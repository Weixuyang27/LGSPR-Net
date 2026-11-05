"""LGSPR-Net with LGSSE, SPS, and PCRA."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


def _group_norm(channels, max_groups=8):
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class Multiscale3DCNN(nn.Module):
    """Local branch of LGSSE."""

    def __init__(self, out_channels=64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1, bias=False),
            _group_norm(16),
            nn.LeakyReLU(inplace=True),
        )
        self.residual = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=3, padding=1, bias=False),
            _group_norm(16),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=3, padding=1, bias=False),
            _group_norm(16),
        )
        self.spectral_conv3 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            _group_norm(16),
            nn.LeakyReLU(inplace=True),
        )
        self.spectral_conv5 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=(5, 1, 1), padding=(2, 0, 0), bias=False),
            _group_norm(16),
            nn.LeakyReLU(inplace=True),
        )
        self.spectral_gate = nn.Sequential(
            nn.Conv3d(
                16, 16, kernel_size=(3, 1, 1), padding=(1, 0, 0),
                groups=16, bias=False,
            ),
            _group_norm(16),
            nn.Sigmoid(),
        )
        self.projection = nn.Conv3d(
            16, out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False
        )
        self.projection_norm = _group_norm(out_channels)

    def forward(self, x):
        first = self.conv1(x)
        feature = self.residual(first) + first
        multiscale = 0.5 * (
            self.spectral_conv3(feature) + self.spectral_conv5(feature)
        )
        gate = self.spectral_gate(feature.mean(dim=(3, 4), keepdim=True))
        feature = feature + multiscale * gate
        feature = F.leaky_relu(
            self.projection_norm(self.projection(feature)), inplace=True
        )
        return feature.mean(dim=2)


class BidirectionalRowColumnSSM(nn.Module):
    """Global LGSSE branch with shared bidirectional row/column Mamba scans."""

    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "LGSPR-Net requires mamba-ssm. Install it before constructing "
                "the proposed model."
            )
        self.row_norm = nn.LayerNorm(dim)
        self.column_norm = nn.LayerNorm(dim)
        self.scanner = Mamba(
            d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.scan_projection = nn.Conv2d(2 * dim, dim, kernel_size=1, bias=False)
        self.local_context = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            _group_norm(dim),
            nn.LeakyReLU(inplace=True),
        )
        self.scan_gate = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=1, bias=False),
            _group_norm(dim),
            nn.Sigmoid(),
        )

    def _bidirectional(self, sequence, norm):
        sequence = norm(sequence)
        forward = self.scanner(sequence)
        backward = torch.flip(
            self.scanner(torch.flip(sequence, dims=[1])), dims=[1]
        )
        return 0.5 * (forward + backward)

    def forward(self, x):
        batch, channels, height, width = x.shape
        local = self.local_context(x)

        rows = x.permute(0, 2, 3, 1).reshape(batch * height, width, channels)
        rows = self._bidirectional(rows, self.row_norm)
        rows = rows.reshape(batch, height, width, channels)
        rows = rows.permute(0, 3, 1, 2).contiguous()

        columns = x.permute(0, 3, 2, 1).reshape(batch * width, height, channels)
        columns = self._bidirectional(columns, self.column_norm)
        columns = columns.reshape(batch, width, height, channels)
        columns = columns.permute(0, 3, 2, 1).contiguous()

        scanned = self.scan_projection(torch.cat((rows, columns), dim=1))
        gate = self.scan_gate(torch.cat((scanned, local), dim=1))
        return gate * scanned + (1.0 - gate) * local


class CrossGuidedFusion(nn.Module):
    """Fuse local and global LGSSE features through channel-wise guidance."""

    def __init__(self, channels):
        super().__init__()
        self.local_norm = nn.LayerNorm(channels)
        self.global_norm = nn.LayerNorm(channels)
        self.query = nn.Linear(channels, channels, bias=False)
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.output = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            _group_norm(channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, local_feature, global_feature):
        batch, channels, _, _ = local_feature.shape
        local_token = self.local_norm(local_feature.flatten(2).mean(2))
        global_token = self.global_norm(global_feature.flatten(2).mean(2))
        attention = torch.softmax(
            self.query(global_token) * self.key(local_token) / math.sqrt(channels),
            dim=-1,
        )
        context = attention * self.value(local_token)
        gate = torch.sigmoid(context + global_token).view(batch, channels, 1, 1)
        return self.output(local_feature * gate + global_feature)


def class_prototypes(features, labels, n_way):
    prototypes = []
    for class_index in range(n_way):
        class_features = features[labels == class_index]
        if class_features.numel() == 0:
            raise ValueError(f"class {class_index} has no support feature")
        prototypes.append(class_features.mean(dim=0))
    return torch.stack(prototypes, dim=0)


def prototype_logits(
    query_features,
    support_features,
    support_labels,
    n_way,
    temperature=0.5,
):
    """Energy-normalized squared-Euclidean prototypical logits."""
    if temperature <= 0:
        raise ValueError("prototype temperature must be positive")
    prototypes = class_prototypes(support_features, support_labels, n_way)
    distances = torch.cdist(query_features, prototypes, p=2.0).pow(2)
    energy = 0.5 * (
        query_features.pow(2).sum(dim=1).mean()
        + prototypes.pow(2).sum(dim=1).mean()
    )
    return -distances / (energy.clamp_min(1e-6) * float(temperature))


class LGSSE(nn.Module):
    """Local--global spectral--spatial encoder used by LGSPR-Net."""

    def __init__(
        self,
        n_bands=102,
        category_dim=128,
        response_dim=32,
        feature_channels=64,
        use_lgsse=True,
    ):
        super().__init__()
        self.num_bands = int(n_bands)
        self.category_dim = int(category_dim)
        self.response_dim = int(response_dim)
        self.use_lgsse = bool(use_lgsse)

        self.local_branch = Multiscale3DCNN(feature_channels)
        if self.use_lgsse:
            self.input_projection = nn.Conv2d(
                self.num_bands, feature_channels, kernel_size=1, bias=False
            )
            self.global_branch = BidirectionalRowColumnSSM(feature_channels)
            self.fusion = CrossGuidedFusion(feature_channels)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        self.category_head = nn.Linear(feature_channels, self.category_dim)
        self.response_head = nn.Linear(feature_channels, self.response_dim)

    def configure_spectral_mappings(self, source_bands, target_bands):
        """Create independent sensor mappings to the shared spectral width."""

        def build_mapping(input_bands):
            if int(input_bands) == self.num_bands:
                return nn.Identity()
            return nn.Sequential(
                nn.Conv2d(
                    int(input_bands), self.num_bands, kernel_size=1, bias=False
                ),
                nn.BatchNorm2d(self.num_bands),
            )

        self.source_mapping = build_mapping(source_bands)
        self.target_mapping = build_mapping(target_bands)
        self.source_input_bands = int(source_bands)
        self.target_input_bands = int(target_bands)

    def map_spectral_input(self, x, domain):
        if domain == "source":
            mapping = getattr(self, "source_mapping", None)
        elif domain == "target":
            mapping = getattr(self, "target_mapping", None)
        else:
            raise ValueError(f"unknown spectral-mapping domain: {domain}")
        return x if mapping is None else mapping(x)

    def feature_map(self, x):
        local = self.local_branch(x.unsqueeze(1))
        if not self.use_lgsse:
            return local
        global_feature = self.global_branch(self.input_projection(x))
        return self.fusion(local, global_feature)

    def forward_embeddings(self, x):
        feature = self.feature_map(x)
        category = self.category_head(self.pool(feature))
        if hasattr(self, "pre_norm_response"):
            early = self.local_branch.conv1[0](x.unsqueeze(1))
            amplitude = early.square().mean(dim=(2, 3, 4)).clamp_min(1e-12).sqrt()
            response = self.response_head(self.pre_norm_response(amplitude))
        else:
            response = self.response_head(self.pool(feature))
        return category, response

    def forward(self, x):
        feature = self.feature_map(x)
        return self.category_head(self.pool(feature))


class LGSPRNet(LGSSE):
    """LGSPR-Net with an LGSSE backbone and training-time SPS/PCRA modules."""

    pass


class _GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.scale * gradient, None


def _reverse_gradient(x, scale=1.0):
    return _GradientReversal.apply(x, scale)


class SPS:
    """Training-only SPS interface."""

    @staticmethod
    def add_arguments(parser):
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--sps", dest="sps", action="store_true")
        group.add_argument("--no_sps", dest="sps", action="store_false")
        parser.set_defaults(sps=True)
        parser.add_argument("--perturb_gain", type=float, default=0.05)
        parser.add_argument("--perturb_spectral_tilt", type=float, default=0.05)
        parser.add_argument("--perturb_offset", type=float, default=0.02)
        parser.add_argument("--lambda_s", type=float, default=0.05)
        parser.add_argument("--lambda_p", type=float, default=0.05)
        parser.add_argument("--sps_warmup", type=int, default=1000)

    @staticmethod
    def configure(model, args):
        model.sps_enabled = bool(args.sps)
        model.sps_stats = {}
        if not model.sps_enabled:
            return
        amplitudes = (
            args.perturb_gain,
            args.perturb_spectral_tilt,
            args.perturb_offset,
        )
        if any(not math.isfinite(value) or not 0 <= value <= 0.1 for value in amplitudes):
            raise ValueError("SPS amplitudes must be finite values in [0, 0.1]")
        if args.perturb_gain <= 0:
            raise ValueError("perturb_gain must be positive")
        if args.sps_warmup < 0 or any(
            not math.isfinite(value) or value < 0
            for value in (args.lambda_s, args.lambda_p)
        ):
            raise ValueError("invalid SPS weights or warm-up")

        device = model.response_head.weight.device
        dtype = model.response_head.weight.dtype
        model.perturbation_predictor = nn.Linear(model.response_dim, 3).to(
            device=device, dtype=dtype
        )
        model.pre_norm_response = nn.Linear(
            model.local_branch.conv1[0].out_channels,
            model.response_head.in_features,
            bias=False,
        ).to(device=device, dtype=dtype)
        model.sps_args = args
        model.sps_scale = 0.0

    @staticmethod
    def set_episode(model, args, episode):
        if model.sps_enabled:
            model.sps_scale = (
                min(1.0, episode / args.sps_warmup)
                if args.sps_warmup
                else 1.0
            )

    @staticmethod
    def perturb(x, args):
        parameters = torch.empty(
            (x.shape[0], 3), device=x.device, dtype=x.dtype
        ).uniform_(-1, 1)
        spectral_axis = torch.linspace(
            -1, 1, x.shape[1], device=x.device, dtype=x.dtype
        ).reshape(1, -1, 1, 1)
        response = (
            1
            + args.perturb_gain * parameters[:, 0, None, None, None]
            + args.perturb_spectral_tilt
            * parameters[:, 1, None, None, None]
            * spectral_axis
        )
        sample_scale = (
            x.detach().square().mean(dim=(1, 2, 3), keepdim=True)
            .clamp_min(1e-12).sqrt()
        )
        changed = (
            x * response
            + args.perturb_offset
            * parameters[:, 2, None, None, None]
            * sample_scale
        )
        return changed, parameters

    @staticmethod
    def loss(model, domains, n_way, temperature, return_augmented=False):
        if not model.sps_enabled:
            zero = domains[0][1].new_zeros(())
            return (zero, None) if return_augmented else zero

        args = model.sps_args
        losses, augmented, diagnostics = [], [], []
        for query_pixels, clean_category, clean_response, support, support_labels, query_labels in domains:
            changed_pixels, parameters = SPS.perturb(
                query_pixels, args
            )
            changed_category, changed_response = model.forward_embeddings(changed_pixels)
            augmented.append(changed_category)

            logits = prototype_logits(
                changed_category, support, support_labels, n_way, temperature
            )
            classification = F.cross_entropy(logits, query_labels)
            stability = (
                F.normalize(changed_category, dim=1)
                - F.normalize(clean_category.detach(), dim=1)
            ).square().sum(dim=1).mean()
            prediction = model.perturbation_predictor(
                changed_response - clean_response
            )
            regression = F.mse_loss(prediction, parameters)
            losses.append(
                0.5 * classification
                + model.sps_scale
                * (args.lambda_s * stability + args.lambda_p * regression)
            )
            diagnostics.append(
                {
                    "aug_ce": float(classification.detach()),
                    "stability": float(stability.detach()),
                    "response_mse": float(regression.detach()),
                }
            )

        result = torch.stack(losses).mean()
        model.sps_stats = {
            "ramp": model.sps_scale,
            "source": diagnostics[0],
            "target": diagnostics[1],
            "weighted_loss": float(result.detach()),
        }
        return (result, augmented) if return_augmented else result

    @staticmethod
    def metadata(args):
        return {
            "sps": bool(args.sps),
            "perturb_gain": args.perturb_gain,
            "perturb_spectral_tilt": args.perturb_spectral_tilt,
            "perturb_offset": args.perturb_offset,
            "lambda_s": args.lambda_s,
            "lambda_p": args.lambda_p,
            "sps_warmup": args.sps_warmup,
        }


class PrototypeConditionedRelation(nn.Module):
    """Relation encoder and classifiers used by PCRA."""

    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2 * dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
        )
        self.pair_classifier = nn.Linear(32, 1)
        self.domain_classifiers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))
                for _ in range(2)
            ]
        )

    def forward(self, query, prototype):
        query = F.normalize(query, dim=1)
        prototype = F.normalize(prototype, dim=1)
        relation = torch.cat(
            ((query - prototype).abs(), query * prototype), dim=1
        )
        return self.encoder(relation)


class PCRA:
    """Training-only PCRA interface."""

    @staticmethod
    def add_arguments(parser):
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--pcra", dest="pcra", action="store_true")
        group.add_argument("--no_pcra", dest="pcra", action="store_false")
        parser.set_defaults(pcra=True)
        parser.add_argument("--lambda_pair", type=float, default=0.1)
        parser.add_argument("--lambda_cd", type=float, default=0.01)
        parser.add_argument("--pcra_start", type=int, default=1000)
        parser.add_argument("--pcra_warmup", type=int, default=1000)
        parser.add_argument("--relation_queries_per_class", type=int, default=2)

    @staticmethod
    def configure(model, args):
        model.pcra_enabled = bool(args.pcra)
        model.pcra_stats = {}
        if not model.pcra_enabled:
            return
        if min(args.pcra_start, args.pcra_warmup) < 0 or any(
            not math.isfinite(value) or value < 0
            for value in (args.lambda_pair, args.lambda_cd)
        ):
            raise ValueError("invalid PCRA schedule or weights")
        if args.relation_queries_per_class < 1:
            raise ValueError("relation_queries_per_class must be positive")

        model.relation_module = PrototypeConditionedRelation(
            model.category_dim
        ).to(
            device=model.category_head.weight.device,
            dtype=model.category_head.weight.dtype,
        )
        model.pcra_args = args
        model.pcra_scale = 0.0

    @staticmethod
    def set_episode(model, args, episode):
        if model.pcra_enabled:
            elapsed = max(0, episode - args.pcra_start)
            model.pcra_scale = (
                min(1.0, elapsed / args.pcra_warmup)
                if args.pcra_warmup
                else float(elapsed > 0)
            )

    @staticmethod
    def pairs(
        support,
        support_labels,
        query,
        query_labels,
        n_way,
        views,
        queries_per_class,
    ):
        if views < 1:
            raise ValueError("views must be positive")
        prototypes = class_prototypes(support, support_labels, n_way)
        real_query = query[::views]
        real_labels = query_labels[::views]
        normalized_prototypes = F.normalize(prototypes, dim=1)

        left, right, relation_labels = [], [], []
        for class_index in range(n_way):
            indices = torch.nonzero(
                real_labels == class_index, as_tuple=False
            ).flatten()[:queries_per_class]
            if indices.numel() == 0:
                raise ValueError("each class needs at least one query")
            for index in indices:
                query_feature = real_query[index]
                distances = (
                    F.normalize(query_feature.unsqueeze(0), dim=1)
                    - normalized_prototypes
                ).square().sum(dim=1)
                distances[class_index] = torch.inf
                hard_negative = int(distances.argmin())
                left.extend((query_feature, query_feature))
                right.extend((prototypes[class_index], prototypes[hard_negative]))
                relation_labels.extend((1.0, 0.0))

        return (
            torch.stack(left),
            torch.stack(right),
            query.new_tensor(relation_labels),
        )

    @staticmethod
    def loss(model, domains, n_way, target_views):
        if not model.pcra_enabled:
            return domains[0][0].new_zeros(())
        if n_way < 2:
            raise ValueError("PCRA requires at least two classes")

        pair_losses, domain_losses = [], []
        pair_accuracy, domain_accuracy = [], []
        for domain_index, (support, support_labels, query, query_labels) in enumerate(domains):
            views = 1 if domain_index == 0 else target_views
            left, right, labels = PCRA.pairs(
                support,
                support_labels,
                query,
                query_labels,
                n_way,
                views,
                model.pcra_args.relation_queries_per_class,
            )
            relation = model.relation_module(left, right)
            pair_logits = model.relation_module.pair_classifier(relation).squeeze(1)
            pair_losses.append(
                F.binary_cross_entropy_with_logits(pair_logits, labels)
            )
            pair_accuracy.append(
                float(((pair_logits > 0) == labels.bool()).float().mean().detach())
            )

            for relation_type in (0, 1):
                selected = relation[labels == relation_type]
                domain_logits = model.relation_module.domain_classifiers[
                    relation_type
                ](_reverse_gradient(selected, model.pcra_scale))
                truth = torch.full(
                    (len(selected),),
                    domain_index,
                    device=relation.device,
                    dtype=torch.long,
                )
                domain_losses.append(F.cross_entropy(domain_logits, truth))
                domain_accuracy.append(
                    float((domain_logits.argmax(1) == truth).float().mean().detach())
                )

        pair_loss = torch.stack(pair_losses).mean()
        domain_loss = torch.stack(domain_losses).mean()
        result = (
            model.pcra_args.lambda_pair * pair_loss
            + model.pcra_args.lambda_cd * domain_loss
        )
        model.pcra_stats = {
            "pair_ce": float(pair_loss.detach()),
            "domain_ce": float(domain_loss.detach()),
            "grl": model.pcra_scale,
            "pair_accuracy": pair_accuracy,
            "domain_accuracy": domain_accuracy,
            "weighted_loss": float(result.detach()),
        }
        return result

    @staticmethod
    def metadata(args):
        return {
            "pcra": bool(args.pcra),
            "lambda_pair": args.lambda_pair,
            "lambda_cd": args.lambda_cd,
            "pcra_start": args.pcra_start,
            "pcra_warmup": args.pcra_warmup,
            "relation_queries_per_class": args.relation_queries_per_class,
        }


__all__ = [
    "LGSPRNet",
    "LGSSE",
    "SPS",
    "PCRA",
    "class_prototypes",
    "prototype_logits",
    "PrototypeConditionedRelation",
]
