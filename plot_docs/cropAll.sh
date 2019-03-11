#!/bin/bash


for filename in ./*.pdf; do
    for ((i=0; i<=3; i++)); do
        pdfcrop "$filename" "$filename"
    done
done

