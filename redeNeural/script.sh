#!/bin/bash
for y in 13 26 39 50; do
    for i in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
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