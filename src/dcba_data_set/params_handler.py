"""A script with functions to facilitate liading simulation's parameters and input data."""

# TODO: review this code and decidee whether it's needed

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import network_diffusion as nd

from dcba_data_set.loaders.constants import SEPARATOR
from dcba_data_set.loaders.net_loader import load_network

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Network:
    """Immutable container pairing a multilayer network with its type and name metadata."""

    n_type: str
    n_name: str
    n_graph_pt: nd.MultilayerNetworkTorch
    n_graph_nx: nd.MultilayerNetwork

    @property
    def rich_name(self) -> str:
        """Return a combined ``type^name`` identifier, or just ``type`` when both are equal."""
        _type = self.n_type.replace("/", ".")
        _name = self.n_name.replace("/", ".")
        if _type == _name:
            return _type
        return f"{_type}{SEPARATOR}{_name}"


def create_out_dir(out_dir: str) -> Path:
    """Create and return the output directory, falling back to a temp dir on collision."""
    try:
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(exist_ok=True, parents=True)
    except FileExistsError:
        logger.warning("Redirecting output to a temporary directory after existing-path collision.")
        out_dir_path = Path(tempfile.mkdtemp())
    return out_dir_path


def load_networks(networks: list[str], device: str) -> list[Network]:
    """Load and convert networks specified by ``type^name`` regex patterns.

    :param networks: List of ``type^name`` strings identifying networks to load.
    :param device: PyTorch device string passed to :func:`nd.MultilayerNetworkTorch.from_mln`.
    :returns: List of populated :class:`Network` instances.
    """
    nets = []
    for net_regex in networks:
        net_type, net_name = net_regex.split(SEPARATOR)
        logger.info("Loading network(s): %s - %s", net_type, net_name)
        for (net_type, net_name), net_graph in load_network(
            net_type=net_type, net_name=net_name
        ).items():
            logger.info("Converting network to PyTorch: %s/%s", net_type, net_name)
            nets.append(
                Network(
                    n_type=net_type,
                    n_name=net_name,
                    n_graph_nx=net_graph,
                    n_graph_pt=nd.MultilayerNetworkTorch.from_mln(net_graph, device),
                )
            )
    logger.info("Loaded %d networks", len(nets))
    return nets
