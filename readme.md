# Documentation

This project provides implementations of various Multi-Armed-Bandit related reinforcement learning algorithms from impactful papers, as well as original recombinations of them.
They have been implemented such that they rely on common structures and common interfaces, allowing easy comparisons in any suited environment and easy extension of various aspects.
Additionally, high big-O runtimes - as present in some pseudocodes intended for theoretical analysis of a strategy's _solution-quality_ only - have been avoided where possible.

## Requirements
- Python 3.8 or higher
- requirements.txt
	```bash
	pip install -r requirements.txt
	```

## Usage

Everything is organized in a highly modular way, meaning there is an individual file for every test, every environment, every learning strategy and every replacable subunit of things.
One file usally contains one class or function that provides some functionality. They can be combined with relative freedom as long as you adhere to their interface expectations outlined in the following section.

Intended points of entry are the scripts in ```tests/```. Any such script can be executed by simply running
```bash
python file.py
```
_from within_ the ```tests/``` folder. The script will then load elements from other folders as specified at its top, assuming the project's root is at ```../``` (hence the place of execution is important).
All scripts outside of ```tests/``` are not meant for direct execution and may not source the root correctly, but this is no issue when scripts that have included them do so because the includes happens at runtime, _after_ a script that includes other things has been included itself.

## Script types and scripts
Now to explain what the scripts in which folder are good for and what the individual scripts are doing.

### tests
As stated before, these scripts serve as the point of entry. Therefore, there are no strict requirements for how they should like. What we usually do is:
- Initiate one or more environments from ```environments/```
- Initiate one or more strategies from ```algorithms/```
- Pass these to the ```test```-function from ```helpers/masterTester```. This will run every strategy against every environment, using the funcion ```testOnly```, and plot the regret. Set the parameter ```purge_existing_results``` to ```False``` if you want to include the results from existing csv-files.
- Alternatively, you can call ```testOnly``` directly. This will only run the tests and write the results to cvs-files without plotting them.
- For plotting results from files, call ```readOneResult``` on every file you want to include. For convenience, you will usually want to use ```readAllResults``` to include all csv-files. Then call ```plotResults``` _once_.
	- The files to not need to be from the same call of ```testOnly```. For example, you might have called ```testOnly``` on some combinations of strategies and environments, but forgotten some. Then you can just call ```testOnly``` on these new ones only and plot all together with ```readAllResults``` and ```plotResults```.
- For every strategy x environment x regretDefinition, one file will be created that includes 2 vectors of vectors: One for the average and one for the standard deviation. This information will go into the filename and is later extracted by ```readOneResult```. In order to plot files from other implementations than this one, you currently have to rename your files accordingly.
	- The standard deviation will be included in the plots in the form of error bars. The error bars will be distributed as evenly as possible, with one always being at the start and one at the end, unless there is only one error bar.. The amount of error bars can be controlled by setting the global variable ERRORBAR_COUNT.
	- The second line can be missing without breaking anything.
- The folder into which the files are written and from which they are read is defined by the global variable ```resultpath```. You may want to include the name of the test scenario and/or a timestamp in that path. If it is not set before first required, it will default to ```results/```.
	- The folder will be created if it does not exist yet. If it does exist, existing files will not be purged before writing new ones, so make sure it does not contain things that mean something entirely different than what you currently testing.

### algorithms
The strategies that can be used in the tests. Each is implemented as a class with at least the following properties:
- ```__init__(self, T):``` The constructor. T is the number of timesteps that the strategy shall spend on every environment.
- ```run(self,env):``` Makes the agent interact with environment env T times, usually learning according to the feedback while doing so. The environment must be suited for the strategy, e.g. a strategy for the standard MAB will not expect the environment to return a list of values as feedback.
- ```get_avg_rgt(self):``` Returns the average regret for every previous timestep of the past run. ```masterTester``` will use this for the plots.
- ```get_cum_rgt(self):``` The same as above, but for cumulative regret. Note that in multi-objective settings, both of them will be based on the instantanious regret intead of what was actually received.
- ```clear(self):``` Resets the object to the state it was in at initialisation.

The following are optional:
- ```get_pto_rgt(self):``` Returns the average pareto regret. Only really makes sense in multi-objective settings.
- ```get_eff_rgt(self):``` Returns the difference in score between the average of what has been achieved so far vs. the theoretical current best available score. Might be negative in certain cases. This intended for multi-objective settings because it takes into account what the strategy actually does.

The algorithms are organized in subfolders:

#### misc

#### modular/selectionModules
Scripts that implement various core strategies about selecting the next action according to what has been observed before.
Their main functionality is to be inquired about what action to take next and then informed about the result. They work event-oriented and all loops accross timesteps and performance evaluations are handled by the caller.
Also note that the module never directly interacts with the environment.

All selection modules provide the following interfaces:
- ```__init__(self,T,num_arm):``` Sets up the module. What arguments are required depends on the module, but the planned runtime and the number of arms are always required unless noted otherwise, so only additional arguments, including hyper-parameters, will be mentioned. Selection modules that operate on multi-objective settings include the history container in which the history will be saved.
- ```suggestArm(self):``` Evaluate the current knowledge according to whatever the underlying strategy is and return what action shall be taken.
- ```thisHappened(self, arm, reward, timestep):``` The module acknowledges what action the agent has actually taken and what the feedback was. Note that this does not have to be the action that the module has suggested; this is useful for expert algorithms if you do not want/need to simulate a feedback that fits the proposal. The timestep is usually inferred by how many times the function has called, but it would be possible to use it.
- ```fullReset(self):``` Fully resets the module to its starting state.
- ```resetArm(self, arm):``` Resets only the collected history about the arm. This function is not supported by all selection modules.

And these are the currently available modules:
- ```AbstractSelectionModule```: Serves as a superclass to most of the selection modules. This avoids redundancy as the using modules are mostly identical except for the actions in the constructor and in ```suggestArm```.
- ```MO_OGDE_Module```: The core logic of the Multi-Objective-Online-Gradient-Decent-with-exploration strategy, see Busa-Fekete et al. (2017): "Multi-objective Bandits: Optimizing the Generalized Gini Index". It requires the number of objectives (=the dimension of one feedback), a learning rate and the gini weights that are globally used to evaluate the costs.
- ```Deltaless_MO_OGDE_Module```: A variant of the ```MO_OGDE_Module``` with a simplified version of learning rate calculation that uses no delta (which in the original version was the probability that certain features cannot be guaranteed to hold).
- ```PL_MO_OGDE_Module```: A variant of the ```Deltaless_MO_OGDE_Module``` that includes a modifier for the learning rate and drops the parameter used for the theoretical guarantees. Intended for expert strategies.
- ```UCBModule```: The core logic of the original UCB strategy, see Auer et al. (2002): "Finite-time Analysis of the Multi-armed Bandit Problem", Machine Learning, 47, pp. 235–256. It requires a parameter that scales how important unexplored potential is. Note that this parameter is hardcoded to values such as 0.5 in some strategies based on UCB, which we did too for those strategies.
- ```UCBForcedExploreModule```: Like UCB, but with forced exploration. The fraction of exploration to be equally distributed across all arms is dictated by a new argument.
- ```Pareto_UCB_Module```: The core logic of the Pareto-UCB strategy, that uses a UCB-like estimation for the pareto front and chooses arms uniformly random from that, see Rezaei Balef and Maghsudi (2023): "Piecewise-Stationary Multi-Objective Multi-Armed Bandit With Application to Joint Communications and Sensing", IEEE WIRELESS COMMUNICATIONS LETTERS, VOL. 12, NO. 5.

#### modular/adaptionModules
These modules implement breakpoint adaption. Just like the selection modules, they are supposed to be informed about what happened at every timestep.
However, they do not propose actions. Instead, they know the selection module that is used alongside them and are allowed to call their resetting function or even outright modify what the selection module believes has happened, usually to reduce the impact of less recent feedback.
This allows almost arbitrary selection modules to have breakpoint adaption without the breakpoint adaption logic being in those modules.
Which in turn saves high amounts of copy-pasted code and makes new recombinations less troublesome.

All adaption modules provide the following interfaces:
- ```__init__(self, selection_module):``` Set up the module. Pass hyper-parameters for the breakpoint adaption logic here.
- ```thisHappened(self, arm, reward, t):``` The module acknowledges what action the agent has actually taken and what the feedback was, *and* it may do something with the selection module if the breakpoint adaption says so.
- ```fullReset(self):``` Resets the module to its original state.

These are the adaption modules:
- ```BOCDModule```: The breakpoint adaption logic from the original Bayesian Online Change-point Detection strategy, see Alami et al. (2020): "Restarted Bayesian Online Change-point Detector achieves Optimal Detection Delay". It calculates its parameters from the number of planned timesteps. It resets an arm in the selection module whenever it believes there has been a breakpoint in that arm.
- ```DiscountModule```: The breakpoint adaption logic from the original Discounted UCB strategy, see Garivier et al. (2008): "On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems". It requires a discount factor that is multiplied onto the history of the selection module in each timestep.
- ```GLRModule```: The breakpoint adaption logic from the original GLR-klUCB strategy, see Besson et al. (2022): "Efficient Change-Point Detection for Tackling Piecewise-Stationary Bandits". It requires a hyper parameter, will reset either one arm or everything when detecting a breakpoint (given by the next argument) and the last argument how many timesteps shall be skipped before running its very costly breakpoint detection again.
- ```MonitorModule```: The breakpoint adaption logic from the original Monitored UCB strategy, see Cao et al. (2019): "Nearly Optimal Adaptive Procedure with Change Detection for Piecewise-Stationary Bandit". It requires a window length which to use for the comparision of more and less recent breakpoint detection and a detection threshold. It will reset everything upon encountering a breakpoint.
- ```NullAdaptionModule```: A module that does nothing. It serves as a stand-in for places that expect any adaption module to exist, but no breakpoint adaption shall be performed because we only want to use the selection module.
- ```SlidingWindowModule```: The breakpoint adaption logic from the original Sliding Window UCB strategy, see Garivier et al. (2008): "On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems". It requires a window length and will remove the impact of any feedback that is older than that from the selection module.

#### modular/moduleUsers
Scripts that combine selection and adaption modules to form actual strategies that satisfy the requirements listed under **algorithms**.
They each will execute all timesteps in a loop. The following happens once per loop:
- Ask the selection module what arm to pick next
- Pick that arm and receive feedback and optimal feedback from the environment
- Tell the modules about the feedback (but not about the optimal one)
- Calculate regret

Note that the multi-objective ones save their history in a ```HistoryContainerMO```. This is so that in expert algorithms, adaption modules can just interact with that constainer and the actions will be relayed to all the experts, and the strategy does not need to hold multiple copies of the same history.

The module using scripts include:
- ```AbstractMAB```: Servers as the superclass to most modular strategies. These strategies differ only in what modules they use and the rest is identical, so all the code that would otherwise be copy-pasted is in ```AbstractMAB``` instead.
- ```BasicMultiObjective```: The original Multi-Objective-Online-Gradient-Decent-with-exploration strategy. It consists of the ```MO_OGDE_Module``` and the ```NullAdaptionModule```.
- ```DeltalessMultiObjective```: A simplified version of ```BasicMultiObjective``` that uses the ```Deltaless_MO_OGDE_Module``` instead of the ```MO_OGDE_Module```.
- ```exptertsMultiObjective```: Meta-learning version of ```BasicMultiObjective```. It uses multiple instances of the ```PL_MO_OGDE_Module``` and tries to listen to the best expert most.
- ```BOCD```: The original Bayesian Online Change-point Detection strategy. It consists of the ```UCBForcedExploreModule``` and the ```BOCDModule```.
- ```BOCD_MO```: A strategy that works for multi-objective settings that have breakpoints. It consists of the ```MO_OGDE_Module``` and the ```BOCDModule```.
- ```discountedMO```: A strategy that works for multi-objective settings that have breakpoints. It consists of the ```MO_OGDE_Module``` and the ```DiscountModule```.
- ```GLR_klUCB```: A simplified version of the GLR_klUCB strategy. It consists of the ```UCBForcedExploreModule``` and the ```GLRModule```.
- ```GLR_MO```: A strategy that works for multi-objective settings that have breakpoints. It consists of the ```MO_OGDE_Module``` and the ```GLRModule```.
- ```MonitoredMO```: A strategy that works for multi-objective settings that have breakpoints. It consists of the ```MO_OGDE_Module``` and the ```MonitorModule```.
- ```MonitoredUCB```: The original monitored UCB strategy. It consists of the ```UCBForcedExploreModule``` and the ```MonitorModule```.
- ```SlidingWindowMO```: A strategy that works for multi-objective settings that have breakpoints. It consists of the ```MO_OGDE_Module``` and the ```SlidingWindowModule```.
- ```ModularUCB```: The original UCB strategy. It consists of the ```UCBModule``` and the ```NullAdaptionModule```.
- ```ParetoUCB```: The original Pareto-UCB strategy. It consists of the ```Pareto_UCB_Module``` and the ```NullAdaptionModule```.
- ```ExpertsMultiObjective```: An expert version of ```BasicMultiObjective```. It runs _multiple_ ```PL_MO_OGDE_Module``` initialised with different learning factors in parallel and learns which one makes the best choices. It includes the ```NullAdaptionModule```.
- ```BOCD_ExpertsMultiObjective```: Like ```ExpertsMultiObjective```, but with the ```BOCDModule``` for breakpoint adaption.
- ```DiscountedExpertsMultiObjective```: Like ```ExpertsMultiObjective```, but with the ```DiscountModule``` for breakpoint adaption.

#### readFM.py
This script will transform a file that includes the log of what music what user has listened to when, into a MO-MAB instance. It will first parse the file linewise and extract the important information. It will pool the times into time slots that have been arbitrarily decided on.
After that, it will infer the most popular tracks and the most active users and, per timeslot, how popular which tracks were for these users.
The resulting bandit problem will have the most popular tracks as arms and the most active users as objectives. Picking some arm is intended to simulate the responses of the different users. Some users can be expected to like certain tracks more than others and some tracks can be expected to be liked by some users more than by others. This is estimated based on the users' track call log.
This can be interpreted as having to provide a mix of tracks that shall reach maximum overall satisfaction among all (most active) users as an online learning problem (even though the original data is offline).

### environments
Environments are what the agents, or rather, the scripts that run the strategies, interact with. They each are initialized with arguments that control how exactly their feedback is generated.
After that, most interaction happens through their ```feedback``` function. This takes the action of the agent and returns an appropiate feedback _and the feedback that the optimal action would have yielded. The latter is used only for calculating the regret, not for the strategies themselves.
Additionally, environments that shift their internal state after a certain number of timesteps provide resetting functionalities.

The following environments are available:
- ```env_stochastic```: The default MAB environment. It has expectation values for each arm and returns these plus the provided noise. It supports settings where the agent can pick k out of n arms.
- ```env_adverse1```: An adversary that rerolls the an arm index for each query and returns 1 for arms with equal or greater that index and 0 for the others.
- ```env_adverse2```: An adversary that rerolls the optimal arm for each query and returns 1 for that and 0 for the others.
- ```env_adverse3```: An adversary that randomly picks an optimal arm after a set number of timesteps have passed. It returns 1 for that arm while the rewards for the other arms sinks exponentially the further their index is away.
- ```env_non_stationary```: The default environment for MAB with breakpoints. Is is provided several sets of distributions for the arms and the timesteps (breakpoints) at which to change to the next set.
- ```EnvParabola```: An environment for online convex optimization strategies. The underlying function is parabola-like and its parameters oscillate slightly after a couple of turns if set to do so. Depending on what the agent is allowed to do, it may return the current function.
- ```EnvMultiOutput```: The default environment for multi-objective MAB, where feedback is multi-dimensional and the agent has to minimize the generalized gini index of the total feedback.
- ```EnvMultiOutputBernoulli```: Like the above, but interprets the provided mu as probabilities and only returns 0 or 1.
- ```EnvMultiOutputNonStationary```: Like ```EnvMultiOutput``` but supports breakpoints like in ```env_non_stationary```. This means that the optimal mix of actions will likely change at fixed timesteps, and this environment will recalculate it then.
- ```EnvMultiOutputRandomized```: Like ```EnvMultiOutput``` but the noise and the expected costs per arm and dimension are uniform drawn randomly _on each reset_.


## Moving files
As there are many cross dependencies, it can be difficult to move files around or rename them (which is more or less the same in terms of impact) without the help of an IDE.
And if a file is renamed and modified at the same time, git will not recognize this because it only sees a tracked file being deleted and a new file with different contents than the deleted one appearing. Therefore it is best to perform this in two seperate commits: One for renaming/moving files and one for fixing the dependencies.
We provide a script for the latter. You just need to write what has been renamed into a file called ```diffs.txt``` and then run the script ```fixRefs.py```. You can make sure to check that it worked correctly by running ```git diff```.

The script expects ```diffs.txt``` to contain the relevant portion of the log for the commit where the renaming took place. If this was the latest commit, you can create and fill the file by running the following command:
```bash
git log --stat -n 1 | grep "=>" | grep "| 0" > diffs.txt
```


# TODO: Insert or remove

## Algorithm
* The Epsilon-Greedy algorithm balances exploration and exploitation. With probability epsilon, the algorithm chooses a random arm (exploration), and with probability 1-epsilon, it chooses the arm with the highest expected reward (exploitation).

* The UCB algorithm  

## Usage
To run the code, simply run `main.py`. You can modify the parameters such as the number of arms, the number of trials, and the epsilon value to experiment with different scenarios.
