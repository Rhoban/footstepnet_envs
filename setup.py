from setuptools import setup

setup(name="gym_footsteps_planning", 
      version="1.0", 
      description='FootStep Planning RL Environment',
      install_requires=[
          "gymnasium==0.29.1",
          "numpy>=1.20.0",
          "stable_baselines3>=2.1.0",
          "sb3-contrib>=2.1.0", 
          "pygame",
          ], 
      )
