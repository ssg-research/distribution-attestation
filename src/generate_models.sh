#! /bin/bash

# Authors: Vasisht Duddu, Anudeep Das, Nora Khayata, Hossein Yalame, Thomas Schneider, N Asokan
# Copyright 2024 Secure Systems Group, University of Waterloo & Aalto University, https://crysp.uwaterloo.ca/research/SSG/
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

python -m src.generate_models --dataset CENSUS --filter sex --split verifier  --ratio 0.0 --num 1000
python -m src.generate_models --dataset CENSUS --filter sex --split verifier  --ratio 0.1 --num 1000
python -m src.generate_models --dataset CENSUS --filter sex --split verifier  --ratio 0.2 --num 1000
python -m src.generate_models --dataset CENSUS --filter sex --split verifier  --ratio 0.3 --num 1000
python -m src.generate_models --dataset CENSUS --filter sex --split verifier  --ratio 0.4 --num 1000
python -m src.generate_models --dataset CENSUS --filter sex --split verifier  --ratio 0.5 --num 1000
python -m src.generate_models --dataset CENSUS --filter sex --split verifier  --ratio 0.6 --num 1000
python -m src.generate_models --dataset CENSUS --filter sex --split verifier  --ratio 0.7 --num 1000
python -m src.generate_models --dataset CENSUS --filter sex --split verifier  --ratio 0.8 --num 1000
python -m src.generate_models --dataset CENSUS --filter sex --split verifier  --ratio 0.9 --num 1000
python -m src.generate_models --dataset CENSUS --filter sex --split verifier  --ratio 1.0 --num 1000


python -m src.generate_models --dataset CENSUS --filter sex --split prover  --ratio 0.0 --num 1000
python -m src.generate_models --dataset CENSUS --filter sex --split prover  --ratio 0.1 --num 1000
python -m src.generate_models --dataset CENSUS --filter sex --split prover  --ratio 0.2 --num 1000
python -m src.generate_models --dataset CENSUS --filter sex --split prover  --ratio 0.3 --num 1000
python -m src.generate_models --dataset CENSUS --filter sex --split prover  --ratio 0.4 --num 1000
python -m src.generate_models --dataset CENSUS --filter sex --split prover  --ratio 0.5 --num 1000
python -m src.generate_models --dataset CENSUS --filter sex --split prover  --ratio 0.6 --num 1000
python -m src.generate_models --dataset CENSUS --filter sex --split prover  --ratio 0.7 --num 1000
python -m src.generate_models --dataset CENSUS --filter sex --split prover  --ratio 0.8 --num 1000
python -m src.generate_models --dataset CENSUS --filter sex --split prover  --ratio 0.9 --num 1000
python -m src.generate_models --dataset CENSUS --filter sex --split prover  --ratio 1.0 --num 1000


python -m src.generate_models --dataset CENSUS --filter race --split verifier  --ratio 0.0 --num 1000
python -m src.generate_models --dataset CENSUS --filter race --split verifier  --ratio 0.1 --num 1000
python -m src.generate_models --dataset CENSUS --filter race --split verifier  --ratio 0.2 --num 1000
python -m src.generate_models --dataset CENSUS --filter race --split verifier  --ratio 0.3 --num 1000
python -m src.generate_models --dataset CENSUS --filter race --split verifier  --ratio 0.4 --num 1000
python -m src.generate_models --dataset CENSUS --filter race --split verifier  --ratio 0.5 --num 1000
python -m src.generate_models --dataset CENSUS --filter race --split verifier  --ratio 0.6 --num 1000
python -m src.generate_models --dataset CENSUS --filter race --split verifier  --ratio 0.7 --num 1000
python -m src.generate_models --dataset CENSUS --filter race --split verifier  --ratio 0.8 --num 1000
python -m src.generate_models --dataset CENSUS --filter race --split verifier  --ratio 0.9 --num 1000
python -m src.generate_models --dataset CENSUS --filter race --split verifier  --ratio 1.0 --num 1000


python -m src.generate_models --dataset CENSUS --filter race --split prover  --ratio 0.0 --num 1000
python -m src.generate_models --dataset CENSUS --filter race --split prover  --ratio 0.1 --num 1000
python -m src.generate_models --dataset CENSUS --filter race --split prover  --ratio 0.2 --num 1000
python -m src.generate_models --dataset CENSUS --filter race --split prover  --ratio 0.3 --num 1000
python -m src.generate_models --dataset CENSUS --filter race --split prover  --ratio 0.4 --num 1000
python -m src.generate_models --dataset CENSUS --filter race --split prover  --ratio 0.5 --num 1000
python -m src.generate_models --dataset CENSUS --filter race --split prover  --ratio 0.6 --num 1000
python -m src.generate_models --dataset CENSUS --filter race --split prover  --ratio 0.7 --num 1000
python -m src.generate_models --dataset CENSUS --filter race --split prover  --ratio 0.8 --num 1000
python -m src.generate_models --dataset CENSUS --filter race --split prover  --ratio 0.9 --num 1000
python -m src.generate_models --dataset CENSUS --filter race --split prover  --ratio 1.0 --num 1000

python -m src.generate_models --dataset BONEAGE  --split verifier  --ratio 0.2 --num 600
python -m src.generate_models --dataset BONEAGE  --split verifier  --ratio 0.3 --num 600
python -m src.generate_models --dataset BONEAGE  --split verifier  --ratio 0.4 --num 600
python -m src.generate_models --dataset BONEAGE  --split verifier  --ratio 0.5 --num 600
python -m src.generate_models --dataset BONEAGE  --split verifier  --ratio 0.6 --num 600
python -m src.generate_models --dataset BONEAGE  --split verifier  --ratio 0.7 --num 600
python -m src.generate_models --dataset BONEAGE  --split verifier  --ratio 0.8 --num 600


python -m src.generate_models --dataset BONEAGE  --split prover  --ratio 0.2 --num 600
python -m src.generate_models --dataset BONEAGE  --split prover  --ratio 0.3 --num 600
python -m src.generate_models --dataset BONEAGE  --split prover  --ratio 0.4 --num 600
python -m src.generate_models --dataset BONEAGE  --split prover  --ratio 0.5 --num 600
python -m src.generate_models --dataset BONEAGE  --split prover  --ratio 0.6 --num 600
python -m src.generate_models --dataset BONEAGE  --split prover  --ratio 0.7 --num 600
python -m src.generate_models --dataset BONEAGE  --split prover  --ratio 0.8 --num 600

python -m src.generate_models --dataset ARXIV --split verifier --degree 9 --num 800
python -m src.generate_models --dataset ARXIV --split verifier --degree 10 --num 800
python -m src.generate_models --dataset ARXIV --split verifier --degree 11 --num 800
python -m src.generate_models --dataset ARXIV --split verifier --degree 12 --num 800
python -m src.generate_models --dataset ARXIV --split verifier --degree 13 --num 800
python -m src.generate_models --dataset ARXIV --split verifier --degree 14 --num 800
python -m src.generate_models --dataset ARXIV --split verifier --degree 15 --num 800
python -m src.generate_models --dataset ARXIV --split verifier --degree 16 --num 800
python -m src.generate_models --dataset ARXIV --split verifier --degree 17 --num 800

python -m src.generate_models --dataset ARXIV --split prover --degree 9 --num 800
python -m src.generate_models --dataset ARXIV --split prover --degree 10 --num 800
python -m src.generate_models --dataset ARXIV --split prover --degree 11 --num 800
python -m src.generate_models --dataset ARXIV --split prover --degree 12 --num 800
python -m src.generate_models --dataset ARXIV --split prover --degree 13 --num 800
python -m src.generate_models --dataset ARXIV --split prover --degree 14 --num 800
python -m src.generate_models --dataset ARXIV --split prover --degree 15 --num 800
python -m src.generate_models --dataset ARXIV --split prover --degree 16 --num 800
python -m src.generate_models --dataset ARXIV --split prover --degree 17 --num 800













python -m src.generate_models --dataset CENSUS --filter sex --split prover  --ratio 0.0 --test --num 1000
python -m src.generate_models --dataset CENSUS --filter sex --split prover  --ratio 0.1 --test --num 1000
python -m src.generate_models --dataset CENSUS --filter sex --split prover  --ratio 0.2 --test --num 1000
python -m src.generate_models --dataset CENSUS --filter sex --split prover  --ratio 0.3 --test --num 1000
python -m src.generate_models --dataset CENSUS --filter sex --split prover  --ratio 0.4 --test --num 1000
python -m src.generate_models --dataset CENSUS --filter sex --split prover  --ratio 0.5 --test --num 1000
python -m src.generate_models --dataset CENSUS --filter sex --split prover  --ratio 0.6 --test --num 1000
python -m src.generate_models --dataset CENSUS --filter sex --split prover  --ratio 0.7 --test --num 1000
python -m src.generate_models --dataset CENSUS --filter sex --split prover  --ratio 0.8 --test --num 1000
python -m src.generate_models --dataset CENSUS --filter sex --split prover  --ratio 0.9 --test --num 1000
python -m src.generate_models --dataset CENSUS --filter sex --split prover  --ratio 1.0 --test --num 1000

python -m src.generate_models --dataset CENSUS --filter race --split prover  --ratio 0.0 --test --num 1000
python -m src.generate_models --dataset CENSUS --filter race --split prover  --ratio 0.1 --test --num 1000
python -m src.generate_models --dataset CENSUS --filter race --split prover  --ratio 0.2 --test --num 1000
python -m src.generate_models --dataset CENSUS --filter race --split prover  --ratio 0.3 --test --num 1000
python -m src.generate_models --dataset CENSUS --filter race --split prover  --ratio 0.4 --test --num 1000
python -m src.generate_models --dataset CENSUS --filter race --split prover  --ratio 0.5 --test --num 1000
python -m src.generate_models --dataset CENSUS --filter race --split prover  --ratio 0.6 --test --num 1000
python -m src.generate_models --dataset CENSUS --filter race --split prover  --ratio 0.7 --test --num 1000
python -m src.generate_models --dataset CENSUS --filter race --split prover  --ratio 0.8 --test --num 1000
python -m src.generate_models --dataset CENSUS --filter race --split prover  --ratio 0.9 --test --num 1000
python -m src.generate_models --dataset CENSUS --filter race --split prover  --ratio 1.0 --test --num 1000

python -m src.generate_models --dataset BONEAGE  --split prover  --ratio 0.2 --test --num 600
python -m src.generate_models --dataset BONEAGE  --split prover  --ratio 0.3 --test --num 600
python -m src.generate_models --dataset BONEAGE  --split prover  --ratio 0.4 --test --num 600
python -m src.generate_models --dataset BONEAGE  --split prover  --ratio 0.5 --test --num 600
python -m src.generate_models --dataset BONEAGE  --split prover  --ratio 0.6 --test --num 600
python -m src.generate_models --dataset BONEAGE  --split prover  --ratio 0.7 --test --num 600
python -m src.generate_models --dataset BONEAGE  --split prover  --ratio 0.8 --test --num 600

python -m src.generate_models --dataset ARXIV --split prover --degree 9 --num 800 --test
python -m src.generate_models --dataset ARXIV --split prover --degree 10 --num 800 --test
python -m src.generate_models --dataset ARXIV --split prover --degree 11 --num 800 --test
python -m src.generate_models --dataset ARXIV --split prover --degree 12 --num 800 --test
python -m src.generate_models --dataset ARXIV --split prover --degree 13 --num 800 --test
python -m src.generate_models --dataset ARXIV --split prover --degree 14 --num 800 --test
python -m src.generate_models --dataset ARXIV --split prover --degree 15 --num 800 --test
python -m src.generate_models --dataset ARXIV --split prover --degree 16 --num 800 --test
python -m src.generate_models --dataset ARXIV --split prover --degree 17 --num 800 --test

