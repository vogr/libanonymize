#!/usr/bin/env bash
set -u

WD="$(dirname "$0")"
FNAME=$(mktemp -p "" "glove.6B.XXXXXX.zip")

wget "http://nlp.stanford.edu/data/glove.6B.zip" -O "$FNAME" &&
unzip -d "$WD" "$FNAME" &&
rm "$FNAME"

