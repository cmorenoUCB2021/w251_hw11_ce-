# w251_hw11_ce-
## w251 - Homework 11

### Carlos Moreno

## (1) What parameters did you change, and what values did you use?

I started by modifying several parameters (one at the time, and combinations).  The main changes were:
- Learning Rate: tried lower values vs. 0.01, but it did not help.
- epsilon decay rate: tried 0.98 (instead of 0.995), but it did not work well.
- numbero of epochs: from 1 to 5 and to 10.

The modifications that worked were the addition of layers to the model.  There were two modifications:
(a) Adding Layers: 64, 32, 16, and 8
(b) Adding layers: 256, 128, 64, 32, 16, and 8

**The results in this report are with the model with six layers (256, 128, 64, 32, 16, and 8).**


## (2) Did you try any other changes (like adding layers or changing the epsilon value) that made things better or worse?

Yes - as discussed above, performance improved mainly when I added layers.  I added 2 and 4 layers, and both models converged.
However, the modifications that used for epsilon decay rate did not work (from 0.995 to 0.98 or 0.998).

The parameters for the best model are:
```
       # Change these parameters to improve performance
        self.density_first_layer = 256      # 64
        self.density_second_layer = 128     # 32
        self.density_third_layer = 64      # 16
        self.density_four_layer = 32        # 8
        self.density_five_layer = 16
        self.density_six_layer = 8

        self.num_epochs = 1
        self.batch_size = 64
        self.epsilon_min = 0.01   #0.01

        # epsilon will randomly choose the next action as either
        # a random action, or the highest scoring predicted action
        self.epsilon = 1.0
        self.epsilon_decay = 0.995   #0.995
        self.gamma = 0.99

        # Learning rate
        self.lr = 0.001
```


## (3) Did your changes improve or degrade the model? How close did you get to a test run with 100% of the scores above 200?

87 out of 100 runs had a score above 200, which represents an Average Reward of 224.952.

```
    Test Performance:
    Average Reward:  224.95232763559798
    Total tests above 200:  87
```
The changes to the architecture (adding layers) improved the model, however the other parameters (i.e. decay, lr) did not have significant impact on model performance.

## (4) Based on what you observed, what conclusions can you draw about the different parameters and their values?

Adding layers improves the ability for the model to learn (reinforce learning) how to land. The other parameters such as learning rate and epsilon_decay seems to align well with the new architecture, and I did not see improvements in performance when modifying them. The learning rate represents how quickly the model learns new Q-value.  In relation to epsiolon and epsilon_decay, at the beginning the model focus on exploration by having a relatively high epsilon. However, as the model learns more, it starts focusing on exploiting the Q-tables (learning) thus, the epsilon_decay helps to reduce the exploration phase so that the model focus on exploitation.

## (5) What is the purpose of the epsilon value?
As we initialize the Q-table with zeros, at the beginning, there is no a best action for the model to take. Thus the model chooses randomly (exploration based on epsilon value). This becomes a challenge once one positive Q-value is found. That leads to the Q-function always returning that specific action (as it was positive). The model may be stuck with this, and it would not explore other avenues, and the model may not be able to explore other avenues to identify an even higher Q-value. In this situation is when the epsilon parameter helps the model. Epsilon helps to decide whether the model is using the Q-function to determine the next action or take a random sample of the action space. Thus the model is not stopping to explore after it finds a positive Q-value (greater zero). Instead the model starts off exploring the action space (randomly) and after every game played it decreases epsilon until reaching minimum (for epsilon). As the model starts with a focus on exploration, it can start learning, and the it can start to focus on exploitation.  This is called the exploration-exploitation trade-off, which is necessary to control the agent’s greed.

## (6) Describe "Q-Learning".

First of all, defining some of the key characters within a reinforced learning scenario:
>- `environment` - situation (i.e. moon) where the agent is located to learn a specific task.
>- `agent` - the object that would be learning a specific task within a given environment (i.e. landing module) 
>- `State` - $s_t$ - this is a specific command for the agent to take within the environment.  The agent receives the `state` and then it takes an `action` (i.e move left, move right).
>- `policy` - the rules that the agent uses to choose an action.
>- `reward` - environment react to an action $A_t$ with a reward $R_{t1}$, and return a new state $S_{t1}$- the new state can be a another command or nothing.

Within a reinforced learning scenario, learning can be modeled using a Q-learning algorithm, which gives the `agent` a memory represented by `Q-table`. In this table of size `states` x `actions`, for each state-action combination a value is storeed. Those values indicate the reward the agent gets by taking an action.  These vales are called Q-values. In summary, `Q-values` represent the “quality” of an action taken from a given state. Higher Q-values imply better chances of getting greater rewards. To calculate them, the following function is used:

$$
Q^{new}(s_t,a_t) = (1 - \alpha) * Q(s_t, a_t) + \alpha * (\tau + \gamma * \text{max}_a Q(s_{t-1},a))
$$

As it can be seen in the formula, the new Q-value of the state-action pair is based on the sum of two parts weighted by a factor ($\alpha$). The first part represents the old Q-value and the second part is the sum of the reward `r` the agent got by taking action at at state $s_t$ and the discounted estimate of optimal future reward. The last term returns the maximum Q-value in the next state $s_{t1}$ over all actions `a`. This represents the future reward, which is discounted by the factor γ. This discount factor is applied as we want the agent to focus more on the immediate rewards while not fully ignoring the future rewards. The parameter α is the learning-rate. It determines to what proportion the model weighs in the two parts into the new Q-value.

As we initialize the Q-table with zeros, at the beginning, there is no a best action for the model to take. Thus the model chooses randomly (exploration based on epsilon value). This becomes a challenge once one positive Q-value is found. That leads to the Q-function always returning that specific action (as it was positive). The model may be stuck with this, and it would not explore other avenues, and the model may not be able to explore other avenues to identify an even higher Q-value. In this situation is when the epsilon parameter helps the model. Epsilon helps to decide whether the model is using the Q-function to determine the next action or take a random sample of the action space. Thus the model is not stopping to explore after it finds a positive Q-value (greater zero). Instead the model starts off exploring the action space (randomly) and after every game played it decreases epsilon until reaching minimum (for epsilon). As the model starts with a focus on exploration, it can start learning, and the it can start to focus on exploitation.  This is called the exploration-exploitation trade-off, which is necessary to control the agent’s greed.

## Apendix: Log for Model Testing

Test Performance:
Average Reward:  224.95232763559798
Total tests above 200:  87

real	11m14.900s
user	0m0.132s
sys	0m0.100s


Lander Testing Full Logs:

0 	: Episode || Reward:  244.09627530616788
1 	: Episode || Reward:  233.06413131453655
2 	: Episode || Reward:  76.3505749048632
3 	: Episode || Reward:  222.21738501357441
4 	: Episode || Reward:  241.3395679076692
5 	: Episode || Reward:  231.65486476177335
6 	: Episode || Reward:  238.72265803609585
7 	: Episode || Reward:  246.5211691323205
8 	: Episode || Reward:  249.4836086977242
9 	: Episode || Reward:  247.66832461499996
10 	: Episode || Reward:  237.35251879956218
11 	: Episode || Reward:  225.96222352279378
12 	: Episode || Reward:  212.23395080464695
13 	: Episode || Reward:  250.97489047973383
14 	: Episode || Reward:  232.333755910778
15 	: Episode || Reward:  224.15277618194241
16 	: Episode || Reward:  239.35664568050487
17 	: Episode || Reward:  243.12624426477365
18 	: Episode || Reward:  233.38816687842248
19 	: Episode || Reward:  259.98861874563136
20 	: Episode || Reward:  283.5983759263657
21 	: Episode || Reward:  242.3119784077172
22 	: Episode || Reward:  230.97756162421277
23 	: Episode || Reward:  194.3341793341997
24 	: Episode || Reward:  252.55037692804444
25 	: Episode || Reward:  226.72895558470577
26 	: Episode || Reward:  240.06847291012028
27 	: Episode || Reward:  244.26179682149228
28 	: Episode || Reward:  229.16428514596413
29 	: Episode || Reward:  205.37026808592418
30 	: Episode || Reward:  221.40438346229138
31 	: Episode || Reward:  246.16420216373427
32 	: Episode || Reward:  128.1090114599066
33 	: Episode || Reward:  203.72763213073088
34 	: Episode || Reward:  239.432173445933
35 	: Episode || Reward:  198.49306513371914
36 	: Episode || Reward:  271.53227591724294
37 	: Episode || Reward:  237.22546170390345
38 	: Episode || Reward:  249.86544377420597
39 	: Episode || Reward:  209.1024432456089
40 	: Episode || Reward:  251.46186695282063
41 	: Episode || Reward:  250.61300240932505
42 	: Episode || Reward:  195.11816836918143
43 	: Episode || Reward:  230.32615886185945
44 	: Episode || Reward:  239.64695343776003
45 	: Episode || Reward:  230.9688330791873
46 	: Episode || Reward:  217.84810576261557
47 	: Episode || Reward:  151.94488814657475
48 	: Episode || Reward:  217.266057071457
49 	: Episode || Reward:  255.44162696647658
50 	: Episode || Reward:  262.56754792203003
51 	: Episode || Reward:  243.38715169316444
52 	: Episode || Reward:  220.7981916680872
53 	: Episode || Reward:  252.51390050066118
54 	: Episode || Reward:  255.7040133306625
55 	: Episode || Reward:  210.4927296687838
56 	: Episode || Reward:  220.28088178318063
57 	: Episode || Reward:  223.90835549069627
58 	: Episode || Reward:  256.2499780954091
59 	: Episode || Reward:  245.97273234475912
60 	: Episode || Reward:  199.89354101944673
61 	: Episode || Reward:  296.01729828236057
62 	: Episode || Reward:  236.15137963468746
63 	: Episode || Reward:  214.6325885252312
64 	: Episode || Reward:  214.75660178614763
65 	: Episode || Reward:  207.99297831982648
66 	: Episode || Reward:  210.78733850913764
67 	: Episode || Reward:  227.5696424884444
68 	: Episode || Reward:  221.9673462667977
69 	: Episode || Reward:  252.51430465939154
70 	: Episode || Reward:  70.89680083818182
71 	: Episode || Reward:  245.09117725184828
72 	: Episode || Reward:  242.32977946146445
73 	: Episode || Reward:  208.59215110529652
^[[B74 	: Episode || Reward:  242.5848319393578
75 	: Episode || Reward:  125.89848672589802
76 	: Episode || Reward:  248.20474624269062
77 	: Episode || Reward:  267.54936462123914
78 	: Episode || Reward:  240.25152734723864
79 	: Episode || Reward:  231.02332217299713
80 	: Episode || Reward:  131.09560038480933
81 	: Episode || Reward:  215.925474822115
^[[A82 	: Episode || Reward:  248.12406588630745
83 	: Episode || Reward:  225.51468981814895
84 	: Episode || Reward:  234.58256657583993
85 	: Episode || Reward:  107.73951383063077
86 	: Episode || Reward:  199.84233965339197
87 	: Episode || Reward:  238.39851756118142
88 	: Episode || Reward:  223.60210872026752
89 	: Episode || Reward:  223.810626333639
90 	: Episode || Reward:  237.8863724666739
91 	: Episode || Reward:  229.08665332217166
92 	: Episode || Reward:  240.35535417701516
93 	: Episode || Reward:  232.33123607902283
94 	: Episode || Reward:  222.69014367544654
95 	: Episode || Reward:  256.59385313546215
96 	: Episode || Reward:  135.23881681631025
97 	: Episode || Reward:  216.23995316095397
98 	: Episode || Reward:  258.00395660111224
99 	: Episode || Reward:  236.54387562238531
Average Reward:  224.95232763559798
Total tests above 200:  87

real	11m14.900s
user	0m0.132s
sys	0m0.100s


-----------------------------------------------------------------------------------------------
2022-03-22 19:03:13.974157: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.10.2
WARNING:tensorflow:Deprecation warnings have been disabled. Set TF_ENABLE_DEPRECATION_WARNINGS=1 to re-enable them.
Using TensorFlow backend.
Starting Testing of the trained model...
2022-03-22 19:03:22.926760: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcuda.so.1
2022-03-22 19:03:22.936327: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1049] ARM64 does not support NUMA - returning NUMA node zero
2022-03-22 19:03:22.936526: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1666] Found device 0 with properties: 
name: NVIDIA Tegra X1 major: 5 minor: 3 memoryClockRate(GHz): 0.9216
pciBusID: 0000:00:00.0
2022-03-22 19:03:22.936738: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.10.2
2022-03-22 19:03:22.936999: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.10
2022-03-22 19:03:22.937164: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcufft.so.10
2022-03-22 19:03:22.938434: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcurand.so.10
2022-03-22 19:03:22.943961: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusolver.so.10
2022-03-22 19:03:22.948630: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusparse.so.10
2022-03-22 19:03:22.948996: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.8
2022-03-22 19:03:22.949513: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1049] ARM64 does not support NUMA - returning NUMA node zero
2022-03-22 19:03:22.950054: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1049] ARM64 does not support NUMA - returning NUMA node zero
2022-03-22 19:03:22.950214: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1794] Adding visible gpu devices: 0
2022-03-22 19:03:22.978343: W tensorflow/core/platform/profile_utils/cpu_utils.cc:98] Failed to find bogomips in /proc/cpuinfo; cannot determine CPU frequency
2022-03-22 19:03:22.979190: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x2175b740 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2022-03-22 19:03:22.979273: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2022-03-22 19:03:23.069365: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1049] ARM64 does not support NUMA - returning NUMA node zero
2022-03-22 19:03:23.069728: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x235875a0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2022-03-22 19:03:23.069825: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): NVIDIA Tegra X1, Compute Capability 5.3
2022-03-22 19:03:23.070569: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1049] ARM64 does not support NUMA - returning NUMA node zero
2022-03-22 19:03:23.070747: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1666] Found device 0 with properties: 
name: NVIDIA Tegra X1 major: 5 minor: 3 memoryClockRate(GHz): 0.9216
pciBusID: 0000:00:00.0
2022-03-22 19:03:23.071081: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.10.2
2022-03-22 19:03:23.071442: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.10
2022-03-22 19:03:23.071572: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcufft.so.10
2022-03-22 19:03:23.071809: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcurand.so.10
2022-03-22 19:03:23.072023: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusolver.so.10
2022-03-22 19:03:23.072301: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusparse.so.10
2022-03-22 19:03:23.072441: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.8
2022-03-22 19:03:23.073085: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1049] ARM64 does not support NUMA - returning NUMA node zero
2022-03-22 19:03:23.073673: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1049] ARM64 does not support NUMA - returning NUMA node zero
2022-03-22 19:03:23.073825: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1794] Adding visible gpu devices: 0
2022-03-22 19:03:23.074210: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.10.2
2022-03-22 19:03:25.267303: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1206] Device interconnect StreamExecutor with strength 1 edge matrix:
2022-03-22 19:03:25.267428: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1212]      0 
2022-03-22 19:03:25.267475: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1225] 0:   N 
2022-03-22 19:03:25.268190: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1049] ARM64 does not support NUMA - returning NUMA node zero
2022-03-22 19:03:25.268867: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1049] ARM64 does not support NUMA - returning NUMA node zero
2022-03-22 19:03:25.269196: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1351] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 155 MB memory) -> physical GPU (device: 0, name: NVIDIA Tegra X1, pci bus id: 0000:00:00.0, compute capability: 5.3)
2022-03-22 19:03:30.578057: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.10




