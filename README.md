# Capture the flag contest (COMP90054 - AI Planning for Autonomy)
The course contest involves a multi-player capture-the-flag variant of Pacman, where agents control both Pacman and ghosts in coordinated team-based strategies.

## Task

This is a **group project** of 3 or 4 members. 

**Your task** is to develop an autonomous Pacman agent team to play the [Pacman Capture the Flag Contest](http://ai.berkeley.edu/contest.html) by suitably modifying file `myTeam.py` (and maybe some other auxiliarly files you may implement). The code submitted should be internally commented at high standards and be error-free and _never crash_. 

In your solution, you have to use at **least 2 AI-related techniques** (**3 techniques at least for groups of 4**) that have been discussed in the subject or explored by you independently, and you can combine them in any form. **We won't accept a final submission with less than 2/3 techniques**. Some candidate techniques that you may consider are:

1. Blind or Heuristic Search Algorithms (using general or Pacman specific heuristic functions).
2. Classical Planning (PDDL and calling a classical planner).
3. Policy iteration or Value Iteration (Model-Based MDP).
4. Monte Carlo Tree Search or UCT (Model-Free MDP).
5. Reinforcement Learning – classical, approximate or deep Q-learning (Model-Free MDP).
6. Goal Recognition techniques (to infer intentions of opponents).
7. Game Theoretic Methods.

We recommend you to start by using search algorithms, given that you already implemented their code in the first project. You can always use hand coded decision trees to express behaviour specific to Pacman, but they won't count as a required technique. You are allowed to express domain knowledge, but remember that we are interested in "autonomy", and hence using techniques that generalise well. The 7 techniques mentioned above can cope with different rules much easier than any decision tree (if-else rules). If you decide to compute a policy, you can save it into a file and load it at the beginning of the game, as you have 15 seconds before every game to perform any pre-computation. If you want to use classical planning, I recommend reading [these tips](CONTEST.md#pac-man-as-classical-planning-with-pddl).

Together with your actual code solution, you will need to develop a **Wiki**, documenting and describing your solution (a 5-min recorded video will also be required, see below).

## Implementation/Solutions Documentation

- [Design Choices](https://abhinavcreed13.github.io/projects/ai-team-pacamon/#design-choices)
  * [Approach One: BFS and A* Heuristic Search](https://abhinavcreed13.github.io/projects/ai-team-pacamon/#approach-one-bfs-and-a-heuristic-search)
  * [Approach Two: Classical Planning](https://abhinavcreed13.github.io/projects/ai-team-pacamon/#approach-two-classical-planning)
  * [Approach Three: Model-based MDP (Value Iteration)](https://abhinavcreed13.github.io/projects/ai-team-pacamon/#approach-three-model-based-mdp-value-iteration)
  * [Approach Four: Monte-Carlo Tree Search](https://abhinavcreed13.github.io/projects/ai-team-pacamon/#approach-four-monte-carlo-tree-search)
  * [Approach Five: Approximate Q-Learning](https://abhinavcreed13.github.io/projects/ai-team-pacamon/#approach-five-approximate-q-learning)
  * [Agent Strategies Improvements](https://abhinavcreed13.github.io/projects/ai-team-pacamon/#agent-strategies-improvements)
- [Experiments and Evaluation](https://abhinavcreed13.github.io/projects/ai-team-pacamon/#experiments-and-evaluation)
  * [Final Submitted Agent](https://abhinavcreed13.github.io/projects/ai-team-pacamon/#final-submitted-agent)
  * [Evolution and Experiments](https://abhinavcreed13.github.io/projects/ai-team-pacamon/#evolution-and-experiments)
- [Conclusions and Reflections](https://abhinavcreed13.github.io/projects/ai-team-pacamon/#conclusions-and-reflections)
- [Youtube Presentation](https://abhinavcreed13.github.io/projects/ai-team-pacamon/#youtube-presentation)

Complete documentation: [https://abhinavcreed13.github.io/projects/ai-team-pacamon](https://abhinavcreed13.github.io/projects/ai-team-pacamon)
 
