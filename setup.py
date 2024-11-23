from setuptools import find_packages, setup

setup(name="gym_footsteps_planning",
    packages=[package for package in find_packages() if package.startswith("gym_footsteps_planning")], 
    version="1.0", 
    description='Footstep Planning RL Environments',
    install_requires=[
          "gymnasium>=0.29.1,<1.1.0",
          "numpy>=1.20.0",
          "stable_baselines3>=2.1.0",
          "sb3-contrib>=2.1.0", 
          "pygame",
          ], 
      )
