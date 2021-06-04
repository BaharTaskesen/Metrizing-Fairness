import os
"""
% Metrizing Fairness
% NeurIPS 2021 Submission
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
This script provides results for Figure~1 and Table~3.
Example usage python run.py
The results are saved under ./results folder.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
if __name__=='__main__':
    # experiments for energy distance
    nlambda = 25
    for dataset in ['Drug', 'CommunitiesCrimeClassification', 'Compas', 'Adult']:# , 'Credit', 'LawSchool']:
        print(dataset)
        print('SGD with Stratified Sampling...')
        for seed in range(10):
            print('Dataset: {}, Seed {}...'.format(dataset, seed))
#             for batchsize in [32, 64, 128]:
            # look at the effect of batch-size in algorithm 2
            print('Running Our Method')
            os.system('python run_benchmark.py --dataset {} --seed {} --a_inside_x True --nlambda {}'.format(dataset, seed, nlambda))
            print('Running FKDE...')
            os.system('python fair_KDE.py --dataset {} --seed {} --nlambda {}'.format(dataset, seed, nlambda))
            if not dataset == 'Adult':
                os.system('python .\MMD_fair_run.py --dataset {} --nlambda {} --a_inside_x True --seed {}'.format(dataset, nlambda, seed))
            
                                                        
#     for dataset in ['CommunitiesCrime', 'StudentsMath', 'StudentsPortugese']:
#         print(dataset)
#         for seed in range(10):
#             print('Seed {}...'.format(seed))
#             os.system('python baseline_convex_fair_regression.py --dataset {} --seed {} --fairness individual'.format(dataset, seed))
        
    
        
# not yet done:
# - wasserstein
# - Experiments with neural network
# - Experiments on Bar-Pass dataset
# - Experiments on more seeds
# - Experiments with psi
        
