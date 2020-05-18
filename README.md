# ReynoldsStressTensor
Demo for publication '5D Flow Tensor MRI to Efficiently Map Reynolds Stresses of Aortic Blood Flow In-Vivo'

Contains a brief demo to illustrate the data evaluation in Walheim, J., Dillinger, H., Gotschy, A. et al. 5D Flow Tensor MRI to Efficiently Map Reynolds Stresses of Aortic Blood Flow In-Vivo. Sci Rep 9, 18794 (2019). https://doi.org/10.1038/s41598-019-55353-x.

Data can be retrieved from https://osf.io/y58ja/. The data used are limited to a systolic frame to reduce the size of the demo code. 
Processing was performed as follows. Reynolds stresses were encoded using a six directional multipoint velocity encoding. For each of the velocity encodings, 5D flow acquisition and reconstruction was performed as in Walheim, J., Dillinger, H. & Kozerke, S. Multipoint 5D flow cardiovascular magnetic resonance - accelerated cardiac- and respiratory-motion resolved mapping of mean and turbulent velocities. J Cardiovasc Magn Reson 21, 42 (2019). https://doi.org/10.1186/s12968-019-0549-0' (reconstruction code to be found in https://osf.io/36gdr/). The reconstruction first reconstructs each velocity encoding separately, enforcing a low-rank model in small image patches (locally low rank) and then combines different encoding strengths for each direction using a Bayesian approach. The final output are estimates of mean velocities and intra-voxel standard deviations of turbulent velocities for each of the encoded directions (results_v and results_std in the following code).