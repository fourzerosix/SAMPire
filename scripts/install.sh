#!/usr/bin/env bash
# SAMPire installer for Linux and macOS. Run from a clone of the repo:  scripts/install.sh
# Creates a private virtualenv (no system Python changes) and links `sampire` into ~/.local/bin.
#   scripts/install.sh --uninstall     remove it again
#   SAMPIRE_YES=1 scripts/install.sh   answer yes to optional installs
set -euo pipefail
cd "$(dirname "$0")/.."

OS="$(uname -s)"
VENV="${SAMPIRE_VENV:-$HOME/.local/share/sampire/venv}"
BIN="$HOME/.local/bin"
say()  { printf '\n==> %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
ask()  { [ "${SAMPIRE_YES:-0}" = "1" ] && return 0; [ -t 0 ] || return 1; read -r -p "$1 [y/N] " a; [[ "$a" =~ ^[Yy] ]]; }

if [ "${1:-}" = "--uninstall" ]; then
  rm -rf "$VENV" "$BIN/sampire"
  echo "Removed $VENV and $BIN/sampire (settings in ~/.config/sampire were kept)."
  exit 0
fi

EXTRA=".[all]"
case "$OS" in
  Linux)
    say "Installing system packages (needs sudo): PortAudio, libsndfile, ALSA headers for MIDI"
    if command -v dnf >/dev/null 2>&1; then
      sudo dnf install -y python3 python3-pip libsndfile alsa-lib-devel gcc python3-devel || warn "dnf had a problem"
      # PortAudio lives in EPEL on RHEL/Rocky 9
      sudo dnf install -y portaudio || { sudo dnf install -y epel-release && sudo dnf install -y portaudio; } || warn "could not install portaudio: audio output will be silent until it is present"
    elif command -v apt-get >/dev/null 2>&1; then
      sudo apt-get update
      sudo apt-get install -y python3 python3-pip python3-venv libportaudio2 libsndfile1 libasound2-dev build-essential python3-dev
    elif command -v pacman >/dev/null 2>&1; then
      sudo pacman -S --needed --noconfirm python python-pip portaudio libsndfile alsa-lib base-devel
    elif command -v zypper >/dev/null 2>&1; then
      sudo zypper install -y python3 python3-pip portaudio libsndfile1 alsa-devel gcc python3-devel
    else
      warn "unknown package manager: install PortAudio, libsndfile and python3-venv yourself"
    fi
    ;;
  Darwin)
    if command -v brew >/dev/null 2>&1; then
      say "Installing PortAudio and libsndfile with Homebrew"
      brew install python portaudio libsndfile || warn "brew install had a problem; continuing"
      export PATH="$(brew --prefix)/bin:$PATH"
    else
      warn "Homebrew not found (https://brew.sh). Continuing with the system Python."
    fi
    ;;
  *)
    warn "unsupported OS '$OS'; trying a generic install"
    EXTRA="."
    ;;
esac

PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 9))' 2>/dev/null; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "Python 3.9 or newer is required." >&2; exit 1; }

say "Creating virtualenv in $VENV using $($PY --version)"
mkdir -p "$(dirname "$VENV")" "$BIN"
"$PY" -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet --upgrade pip setuptools wheel
say "Installing SAMPire ($EXTRA)"
if ! "$VENV/bin/python" -m pip install --upgrade "$EXTRA"; then
  warn "optional extras failed (usually python-rtmidi or aubio); retrying with audio and midi"
  "$VENV/bin/python" -m pip install --upgrade ".[audio,midi]" || "$VENV/bin/python" -m pip install --upgrade ".[audio]"
fi
ln -sf "$VENV/bin/sampire" "$BIN/sampire"

case ":$PATH:" in
  *":$BIN:"*) ;;
  *) warn "$BIN is not on your PATH; add:  export PATH=\"$BIN:\$PATH\"" ;;
esac
say "Verifying"
"$VENV/bin/sampire" --selftest | tail -3
echo
echo "Done. Run:  sampire --demo        (no audio files needed)"
echo "      or:   sampire loop.wav      (chop your own audio)"
