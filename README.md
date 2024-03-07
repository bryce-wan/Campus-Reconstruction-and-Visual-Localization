# Campus-Reconstruction-and-Visual-Localization

## Introduction

---

Term project of course Introduction to Computer Vision (2023-2024) in Zhejiang University

## Description

In this project, the aim is to reconstruct a building in our campus using SfM and then complete the visual localization task based on query images (to get a 6 DOF camera pose with the 3D model reconstructed and the query image). I first took a large number of photos of a building on campus during the day and used these photos to reconstruct a 3D sparse model of the building using [hloc](https://github.com/cvg/Hierarchical-Localization) (a hierarchical localization toolbox). During the reconstruction process, I used several different algorithms for the reconstruction and compared the results. Next, I took many photos of the building at night with low brightness and used these photos for visual localization. I did this first using the localization algorithms already implemented in hloc, and then I trained our own neural network based on [pixloc](https://github.com/cvg/pixloc) for end-to-end visual localization testing. Ultimately, we compared these two visual localization schemes (mainly measured by the mean reprojection error).

## Demo

Click here to see the [demo](https://bryce-wan.github.io/2024-01-09/3D-reconstruction-and-visual-localization-of-the-campus)