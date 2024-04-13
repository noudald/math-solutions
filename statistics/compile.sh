#!/bin/bash

while inotifywait -e close_write --format "*.tex" .; do
    pdflatex main.tex;
done
