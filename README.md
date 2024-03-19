

# Attesting Distributional Properties of Training Data for Machine Learning

This repository contains the code for the experiments used in the paper "Attesting Distributional Properties of Training Data for Machine Learning", which will appear in ESORICS 2024.

## Environment Setup

You need __conda__. Create a virtual environment and install requirements:

```bash
conda env create -f environment.yml
```

To activate:

```bash
conda activate property-attestation
```

To update the env:

```bash
conda env update --name property-attestation --file environment.yml
```

or

```bash
conda activate property-attestation
conda env update --file environment.yml
```

## Dataset and Models

Download and extract the dataset and place it in the main directory.

Link to datasets: https://drive.google.com/file/d/1htV_ssC-eleYoX4144ywWOPzwvvRo0By/view?usp=sharing

Download and extract the trained models and place it in the main directory.

Link to models: https://drive.google.com/file/d/1amm-MXcKF4jKmwoNqjn_PFzGDySgEtSO/view?usp=sharing

## Usage

### Generate models trained on different ratios of sensitive attribute in training data

First step is to train models with different ratios.

```bash
python -m src.generate_models --dataset {CENSUS, BONEAGE, ARXIV} --ratio {} --split {prover,verifier} --filter {} {--use_single_attest} --num 100
```

Filter is ''sex'' or ''race'' for CENSUS. No filter parameter for BONEAGE or ARXIV.
For ARXIV, use ''degree'' instead of ''ratio'' as this is the property of interest for ARXIV.
''--use_single_attest'' argument is needed for CENSUS
Supported ratios/degrees for different datasets are:
```bash
'CENSUS': ["0.0", "0.1", "0.2", "0.3","0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
'BONEAGE': ["0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8"]
'ARXIV': ["9", "10", "11", "12", "13", "14", "15", "16", "17"]
```

To generate all models automatically, you can run a script:
```bash
chmod +x generate_models.sh
./generate_models
```



## Property Attestation

For training and evaluating the effectiveness of property attestation, running on only on the best out of 1 representation generators,

```bash
python -m src.train_verifier_model --dataset {CENSUS,BONEAGE,ARXIV} --filter {sex,race} --claim_ratio {} --ntimes 1
```
By default, ''ntimes'' is set to 10; the best representation generator out of 10 is saved.

Filter is ''sex'' or ''race'' for CENSUS. No filter parameter for BONEAGE OR ARXIV.
For ARXIV, use ''degree'' instead of ''ratio'' as this is the property of interest for ARXIV.

This will train, a model, save it in the specified --classifier-save-path directory (default: "test/"), and also evaluate.

To run training and testing for all supported properties on a specific dataset, run the appropriate script as specified below.

```bash
./src/launch_arxiv_tests.sh
./src/launch_census_tests.sh
./src/launch_boneage_tests.sh
```

### Parsing logs
After running one the above scripts, logs will be made in the src/ directory.
You can parse the logs with the following

```bash
python -m src.parse_logs --dataset {CENSUS,BONEAGE,ARXIV} --claim_property {} --filter {} --logfile_suffix test --dataset_suffix ''
```

''claim_property'' replaces the name of claim_degree and claim_ratio; separate arguments do not need to be used for claim_ratio and claim_degree.
''filter'' is ''sex'' or ''race'' for CENSUS. No filter parameter for BONEAGE or ARXIV

''logfile_suffix'' specifies the last "word" before ''.log'' in the log file names, where a "word" is the snake-case segment of text following the property value. As can be seen by the launch scripts, a default logfile_suffix is 
"test_adv_train"- the corresponding log file names are of the format "<DATASET NAME>_bb_<property>_test_adv_train.log".

''dataset_suffix'' specifies the the snake-case text segment following the dataset name where the plots for a particular dataset are stored. By default, these are stored in plots/<DATASET NAME> (which has a suffix of ''), but there may be a suffix after <DATASET NAME> (eg. <DATASET NAME>_defended, where the suffix is "_defended").

These will make a table containing the results for a specific claim property value. To get results for all supported ratios or degrees for a specified dataset, run the following

```bash
python -m src.assemble_final_attestation_tables --dataset {CENSUS,BONEAGE,ARXIV} --claim_property {} --filter {} --logfile_suffix test --dataset_suffix ''
```

Both of these scripts will save the tables in the plots/ directory for the specific dataset.

### Attestation Evaluation
For evaluation only, use
```bash
python -m src.verification --dataset {CENSUS,BONEAGE,ARXIV} --filter {sex,race} --claim_ratio/claim_degree {}
```
Specify a --classifier-save-path directory if the default was not used for train_verifier_model.


## Robustness
In robustness experiments, we assume a malicious prover which aims to fool the verifier into wrongfully accepting a model with property p != p_claim. The prover trains their own attestation classifier for p_claim, and uses gradients to generate adversarial perturbations of their base model's first layer weights. The prover will then finetune this basemodel, and give it to the verifier.

In order for a verifier to defend against this attack, they must generate adversarially perturbed first layer weights and use these perturbed representations, along with clean representations, in order to train a new property attestation classifier. 


### Conducting Adversarial Attacks

The primary steps for these experiments include:

1. Training the verifier's attestation classifier as usual (the scripts and commands have already been specified above). Please add a suffix for the saved attestation classifier in the tests/<DATASET NAME> directory. Step 3 will assume that these attestation classifiers were saved in test/ARXIV_win1 (ie. --classifier_save_suffix of "win_1" was used in src.train_verifier_model invocation)

2. Training the prover's attestation classifier. Please add a suffix for the saved attesation classifier. Step 3 will assume that the prover's attestation classifier was saved in test/ARXIV_prover_win1. An example is below

```bash
python -m src.train_verifier_model --dataset ARXIV --claim_degree 17 --classifier_save_suffix _prover_win1 --train_prover_version >> src/ARXIV_17_test_prover.log
```
The key argument to note for this case is the ''--train_prover_version'' argument which uses the prover's split for training the attestation classifier.

3. Host an adversarial attack (default: blackbox) against the verifier's undefended attestation classifier. The following is an example script.

```bash
python -m src.verification --dataset ARXIV --classifier_suffix _win1 --do_attack --bb_adv_attack --classifier_suffix_prover _prover_win1 --claim_degree 17 > ARXIV_bb_17_adv_vuln_e8255.log
```
''classifier_suffix'' specifies the the snake-case text segment following the dataset name where the verifier's attestation classifiers for a particular dataset are stored. By default, these are stored in test/<DATASET NAME> (which has a suffix of ''), but there may be a suffix after <DATASET NAME> (eg. <DATASET NAME>_defended, where the suffix is "_defended").

''classifier_suffix_prover'' specifies the classifier suffix for the prover's attestation classifiers. 

''bb_adv_attack'' specifies that this attack is blackbox. This means that the adversarial samples are going to be generated from the prover's attestation classifier which is different than the verifier's attestation classifier. If this flag is set, the classifier_suffix_prover must be different from the classifier_suffix. Furthermore, if this flag is not set, then adversarial samples will be generated from the verifier's attestation classifier in a whitebox attack. 

Note also that the attacks are targeted attacks by default. This means that the prover aims to fool the verifier into accepting their model. In an untargeted attack, the prover aims to do this, and also fool the verifier into rejecting models that should have been accepted. Specify the ''untargeted_attack'' flag in order to conduct an untargeted attack. 


### Adversarial training

In order to conduct adversarial training, at least step 1 of the above steps must be conducted. ie. The verifier must have an attestation classifier. Afterwards, they must generate the adversarial samples which will be used for adversarial training. Note that all scripts running src.verification.py are named in the format "launch_bb_attacks_<dataset name>.sh" files, and all scripts running src.train_verifier_model.py are named in the format "launch_<dataset name>_tests.sh".

The primary steps for adversarial training are as follows:

1. Generate the perturbed samples

The following a sample script of how this is conducted. Note the ''do_defense'' flag has been set.
```bash
python -m src.verification --dataset BONEAGE --classifier_suffix _preFeb_win1 --do_defense --adv_samples_path adv_samples_bb/ --claim_ratio 0.6 > BONEAGE_bb_0.6_adv_gen_e8255.log
```

''adv_samples_path'' specifies the directory where the adversarial samples will be stored. 

This allows the verifier to conduct a whitebox attack against their own attestation classifier and generate adversarially perturbed first layer weights.

Note that if a classifier_suffix_prover, and bb_adv_attack arguments are set, they will have no effect in this case since the do_attack flag is not set. Note also, the the do_attack and do_defense flags should not both be set at the same time. 


2. Adversarially train an attestation classifier.

A sample log looks as follows. 
```bash
python -m src.train_verifier_model --dataset CENSUS --filter race --classifier_save_suffix _defended_fix --claim_ratio 0.9 --do_defense --adv_samples_path adv_samples_bb/ >> src/CENSUS_r_09_test_adv_train.log
```
Note the do_defense flag has been set. The program expects the perturbed first layer weights to be stored in the directory specified by the ''adv_samples_path'' argument, as previously. Remember to add a suffix via --classifier_save_suffix to the test/CENSUS directory where the attestation classifier will be stored. 


3. Conduct adversarial attack (default: blackbox) against the adversarially trained attestation classifier.
The following is a sample script.

```bash
python -m src.verification --dataset ARXIV --classifier_suffix _defended_fix --do_attack  --test_defended --claim_degree 17 > ARXIV_bb_17_adv_defended_e8255.log
```

The ''classifier_suffix'' provided is now the classifier suffix for the verifier's adversarially trained attestation classifier. 
The ''test_defended'' argument is not mandatory.

For all of the above robustness commands, please uncomment the relevant lines in launch_bb_attacks_<dataset name>.sh and launch_<dataset name>_tests.sh in order to run them.


### Base model finetuning

When the adversarially perturbed first layer weights are naively inserted back into the base models, the accuracy of the model worsens. This necessitates finetuning of the base model after insertion, wherein the inserted weights are frozen and all of the other weights are finetuned. With the following commands, the original base model performance, the base model performance after re-insertion and before finetuning, and the performance after finetuning can all be analyzed.

The following is a sample script of how this is conducted. 

```bash
python -m src.verification --dataset ARXIV --classifier_suffix _defended_fix --do_attack  --test_defended --check_finetuned_accuracy --adv_basemodels_path 'results_adv_bb/'--claim_degree 17 --num_finetuned_basemodels 10 > ARXIV_bb_17_adv_defended_e8255.log
```

Note the addition of the ''check_finetuned_accuracy'' flag in particular. The ''adv_basemodels_path'' denote the directory where the finetuned basemodels will be stored, and the ''num_finetuned_basemodels'' argument denotes the number of base models to finetune per ratio. Defaults are 'results_adv_bb/' and 10 respectively. 

The resulting logs can be parsed in the following command to see the statistics for the accuracies. Note the ''finetuned_performance_check'' flag.

```bash
python -m src.parse_logs --dataset ARXIV --claim_property 17 --logfile_suffix _adv_defended_e8255 --finetuned_performance_check >> ARXIV_finetuned_checks.log
```

The finetuned_check.sh script can run all of these. 

## Generate Effective FAR and Cost Data

In order to generate the effects FAR and costs plots and logs, there are (up to) 3 scripts that need to be run. 

If you have logs from running inference already (ie. from "launch_bb_attacks_<dataset>.sh"), then you will only need to run two scripts. Otherwise, please run the launch_bb_attacks_<dataset>.sh scripts for the dataset of your choice. Below is a sample line from this script:

```bash
python -m src.verification --dataset ARXIV --classifier_suffix _win1   --claim_degree 17 > ARXIV_bb_17_baseline.log
```

In this line, we assumed that the ARXIV attestation classifiers are stored in test/ARXIV_win1. If you wish to invoke this test against adversarially trained classifiers, please ensure that you use the correct classifier_suffix which holds the adversarially trained models. Add the --do_attack argument if you desire to do an attack as well, as per the adversarial training instructions above.

Next, create a directory in src/ and place the logs generated from the above script into this directory. To agree with the script,
let the name of this directory be src/baseline_inf_costs.

The second script that must be run is generate_costs_data.sh. This will parse the logs placed in src/baseline_inf_costs and generate further logs in src/hybrid_results.

Finally, run gather_finetuned_far_cost_per_ds.sh. This will parse through the logs in src/hybrid_results and plot the FAR and cost
data located therein.


## Analyzing Hybrid Costs

In order to generate the hybrid inference cost logs and data, please run generate_costs_data.sh. 
The entries in generate_costs_data.sh look as follows:

```bash
python -m src.parse_logs --logfile_suffix "adv_defended" --logfile_subdir "baseline_defended_aug8" --dataset CENSUS --filter sex --hybrid_analysis --parse_test --claim_property 0.0 > CENSUS00_s_hybrid_analysis_plotting_test.log
```

The above line assumes that all of the inference logs (from running src.verification) are located in 
"baseline_defended_aug8", and the log files' names are in the form <DATASET>_<filter>_bb_<ratio>_baseline.log (notably the baseline follows after the "<ratio>_"). If the inference logs are located elsewhere, please update the script accordingly.

The --parse_test argument is used only when you wish to extract the FAR/TAR/TA/FA/cost/ etc. information for the verifier's test dataset. Otherwise, by default, the script is run against the verifier's verification dataset. 

The resulting logs from this script can be moved to another directory. This script also puts the generated plots in src/cost_far_plots, and it produces pkl files holding the FAR vs Expected cost information for use later on. 

### p_req vs cost plots
To generate the p_req vs cost plots, please use the quick_parse_far_frr.sh script. This script goes through each of the hybrid_analysis_plotting logs generated from the above script, and prints out the desired information as denoted by the grep.
Update the directory, logfile pattern, and grep as needed. The script will print out the desired values.

After the desired values from the above script is printed out, please fill out the corresponding lists in src/expected_overheads.py. Adjust the arguments as needed and the p_req vs cost plots will be created.

### far vs cost plots

Assuming the pkl files from generate_costs_data.sh are located in your current directory, please run src/far_cost_graph.py,
specifying the arguments as needed. For this Python script, the default analysis is done on the verifier's verification dataset (ie. the pkl files that end in *verify.pkl). To change this to the verifier's test dataset, please use the --test argument. 


After this, the generated logs can be moved to src/hybrid_results, and the gather_finetuned_far_cost_per_ds.sh script can be optionally run. 


## Credits

Code adapted from paper "Formalizing and Estimating Distribution Inference Risks": https://github.com/iamgroot42/FormEstDistRisks

To run the original whitebox metaclassifier based property inference attack:

```bash
python -m src.property_inference_attack --dataset {CENSUS,BONEAGE} --filter {sex,race} --claim_ratio {}
```
Filter is ''sex'' or ''race'' for CENSUS. No filter parameter for BONEAGE.