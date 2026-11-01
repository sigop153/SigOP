"""Confine the proposer session with bubblewrap.

The proposer is a coding agent with a shell, so the only reliable boundary is a mount
namespace that does not contain what it must not see. Inside the sandbox it gets:

    its run folder                  read-write; the only writable place that persists
    sigop/ (this package)           read-only, so `from sigop.harness import ...` works
    the Python environment          read-only
    the claude binary, /usr, /etc   read-only
    /tmp and $HOME                  fresh tmpfs, gone when the session ends
    the network                     shared, to reach the proxy and the executor

Nothing else is mounted: not the heldout folder, not objectives/, not other runs, not the
user's home. A read of any of them fails with "no such file". Its own PID namespace keeps a
stray kill from reaching the loop or the servers.

This is the wrapper the paper's runs used, without the login and hook plumbing.
"""
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

CLI_HOME = "/mh_cli_home"          # where a session's config dir is mounted inside the sandbox


@dataclass(frozen=True)
class Sandbox:
    run_root: Path            # bound read-write
    package: Path             # the sigop/ folder, bound read-only at its own path
    environments: tuple[Path, ...]   # Python prefixes bound read-only (the venv, its base)
    binary: Path              # the real claude executable
    home: Path                # path of the tmpfs home inside the sandbox
    credentials: Path | None = None    # the CLI's login file, bound read-write for an Anthropic proposer
    dns: bool = False                  # bind the resolver's state, so api.anthropic.com resolves

    def command(self, inner: list[str], config_dir: Path | None = None) -> list[str]:
        """config_dir: a Claude Code config home bound at CLI_HOME, with CLAUDE_CONFIG_DIR set to it."""
        bwrap = shutil.which("bwrap")
        if bwrap is None:
            raise SystemExit("bubblewrap is not installed (apt install bubblewrap), and the proposer needs it to be "
                             "confined; pass --sandbox false to run it unconfined")
        parts = [bwrap, "--die-with-parent", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
                 "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                 "--ro-bind", "/usr", "/usr", "--ro-bind", "/etc", "/etc"]
        for top in ("bin", "sbin", "lib", "lib64"):
            path = Path("/", top)
            if path.is_symlink():
                parts += ["--symlink", os.readlink(path), str(path)]
            elif path.is_dir():
                parts += ["--ro-bind", str(path), str(path)]
        parts += ["--tmpfs", str(self.home), "--ro-bind", str(self.binary), str(self.binary)]
        if self.credentials:
            # The CLI rewrites this file in place on token refresh, so it is the real file,
            # not a copy, and read-write
            parts += ["--bind", str(self.credentials), str(self.credentials)]
        if self.dns and Path("/run/systemd/resolve").is_dir():
            parts += ["--ro-bind", "/run/systemd/resolve", "/run/systemd/resolve"]
        for prefix in self.environments:
            parts += ["--ro-bind", str(prefix), str(prefix)]
        parts += ["--ro-bind", str(self.package), str(self.package),
                  "--bind", str(self.run_root), str(self.run_root),
                  "--chdir", str(self.run_root),
                  "--setenv", "HOME", str(self.home), "--setenv", "TMPDIR", "/tmp"]
        if config_dir:
            parts += ["--bind", str(config_dir), CLI_HOME, "--setenv", "CLAUDE_CONFIG_DIR", CLI_HOME]
        parts += ["--", *inner]
        return parts


def default_sandbox(run_root: Path, binary: str = "claude", anthropic: bool = False) -> Sandbox:
    """The sandbox for this checkout and this Python. With anthropic, the CLI's login file is
    bound in (when there is one; an ANTHROPIC_API_KEY in the environment works without it)
    and DNS is reachable."""
    real = shutil.which(binary)
    if real is None:
        raise SystemExit(f"the {binary} CLI is not on PATH")
    credentials = Path.home() / ".claude" / ".credentials.json"
    return Sandbox(run_root=run_root.resolve(), package=Path(__file__).resolve().parent,
                   environments=python_prefixes(), binary=Path(real).resolve(), home=Path.home(),
                   credentials=credentials if anthropic and credentials.exists() else None, dns=anthropic)


def python_prefixes() -> tuple[Path, ...]:
    """Every directory this interpreter is reached through: the venv, its base, and each
    hop of the executable's symlink chain (a uv-managed Python sits behind a version-less
    symlink folder, and the venv's python points at that, so the hop must exist too)."""
    prefixes = {Path(sys.prefix), Path(sys.base_prefix)}
    hop = Path(sys.executable)
    while hop.is_symlink():
        hop = hop.parent / os.readlink(hop)
        prefixes.add(hop.parent.parent)          # <prefix>/bin/python -> <prefix>
    return tuple(sorted(prefixes))
