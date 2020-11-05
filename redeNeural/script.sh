#!/bin/bash
for y in 26; do
    for i in 0.9; do
        echo "Energia npy"
        python ./energiaSimples/energiaSimplesJuliano.py $i $y
        echo "Energia Categorias"
        python ./energiaSimples/energiaCategoriasSimples.py $i $y
        echo "Rnn"
        python ./rnns/rnn.py $i $y
        echo "RunModel"
        python ./runModels/runModelSemVoto.py $i $y
    done
done