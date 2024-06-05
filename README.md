# For instruction, please visit notebook
https://colab.research.google.com/drive/17_CGR6j1zOVGyHkitFmRiV1BCOnCd_gh?usp=sharing

# How to training with this Q - learning version:
+ rechange the caro size in "caro_cpu.py" to the size you expect.
+ Select an opponent for the agent: adding opponent to the directory, change the agent's opponent at cpu_test_local.
+ let train the model by run "cpu_local_test.py"


# How to check the training results of the model
+ If you wanna view the result with pretrain 5x5 model, download q-table5 from this link and add to directory: https://drive.google.com/drive/folders/13WNqPKLxsASW4koeWXIJ6LwsZGFfdCK-?usp=sharing
+ The policy-off will be saved in q -table of corresponding size
+ The win rate will  be saved to winrate file of corresponding size
+ The effectiveness of the model will be displayed immediately during the training process,
but to show it more clearly, change the epsilon at q_cpu_no_heurstic to 0 and run the test again.