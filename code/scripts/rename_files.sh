#!/bin/bash

DIR="$1"


if [ -z "$DIR" ] || [ ! -d "$DIR" ]; then
    echo "Por favor, especifica un directorio válido."
    exit 1
fi


for file in "$DIR"/*; do

    [ -f "$file" ] || continue


    filename=$(basename "$file")
    ext="${filename##*.}"      
    base="${filename%%_*}"    


    newname="$DIR/$base.$ext"


    mv "$file" "$newname"
    echo "Renombrado: $filename → $base.$ext"
done
