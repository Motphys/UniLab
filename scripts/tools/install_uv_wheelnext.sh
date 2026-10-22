#!/bin/sh
set -eu

VERSION="0.9.14"
BASE_URL="https://wheelnext.astral.sh/v0.0.3"
DESTINATION="${1:-$HOME/.local/bin/uv-wheelnext}"

case "$(uname -s)/$(uname -m)" in
    Linux/x86_64)
        ARTIFACT="uv-x86_64-unknown-linux-gnu.tar.gz"
        SHA256="0d53bc3a68ce43551495477c427205455c13d9fc658d1d971630b1191dfaa8cf"
        ;;
    Linux/aarch64)
        ARTIFACT="uv-aarch64-unknown-linux-gnu.tar.gz"
        SHA256="d0a9f01a4ecdf1fe780963ad5d519a8973a3385c18697e63353966804bed3547"
        ;;
    Darwin/arm64)
        ARTIFACT="uv-aarch64-apple-darwin.tar.gz"
        SHA256="a270f66fc75bf90bf1dd204c1a4c97c2c4dbb421db382fc23873f576d79030ec"
        ;;
    Darwin/x86_64)
        ARTIFACT="uv-x86_64-apple-darwin.tar.gz"
        SHA256="ffd6a326e7d42bda52d36d57e32eef7ea5ade316a4730320ed715e05b16f0cb8"
        ;;
    *)
        echo "Unsupported platform for uv-wheelnext: $(uname -s)/$(uname -m)" >&2
        exit 2
        ;;
esac

download_dir="$(mktemp -d)"
archive="$download_dir/$ARTIFACT"
staging="$(mktemp -d)"
trap 'rm -rf "$download_dir" "$staging"' EXIT
curl --fail --location --silent --show-error --output "$archive" "$BASE_URL/$ARTIFACT"

if command -v sha256sum >/dev/null 2>&1; then
    echo "$SHA256  $archive" | sha256sum --check --
else
    echo "$SHA256  $archive" | shasum -a 256 --check -
fi

tar -xzf "$archive" -C "$staging" --strip-components=1
mkdir -p "$DESTINATION"
cp "$staging/uv" "$staging/uvx" "$DESTINATION/"
"$DESTINATION/uv" --version
printf 'Installed uv-wheelnext %s to %s\n' "$VERSION" "$DESTINATION"
