# SVD SIREN / Joint Neural Representation for Multiple Light Fields

This code describes and allows for the execution of the algorithm described in the paper "Joint Neural Representation for Multiple Light Fields" submitted to ICASSP 2023.
It it designed to learn a joint implicit neural representation of a collection of light fields.

The algorithm works by learning a factorisation of the weights and biases of a SIREN (Sitzmann et al. [2020](https://dl.acm.org/doi/10.5555/3495724.3496350)) based network. The network itself is inspired from the SIGNET architecture (Feng & Varshney [2021](https://ieeexplore.ieee.org/document/9710101)), a variant of SIREN. However, unlike SIGNET which uses the Legendre polynomials as the positional encoding, the model used in the described here is using learned Fourier Features (Tancik et al. [2020](https://bmild.github.io/fourfeat/)).

For each layer, a base of matrices is learned that serve as a representation shared between all light fields (a.k.a. scene) of the dataset, together with, for each scene in the dataset, a set of coefficients with respect to this base, which act as individual representations. 

The matrices formed by taking the linear combinations of the base matrices with the coefficients corresponding to a given scene serve as the weight matrices of a SIREN network.
In additional to the set of coefficients, we also learn an individual bias vector for each scene.

More precisely, the code uses matrix factorization inspired from the Singluar Value Decomposition (SVD). For each layer with input size `fan_in` and output size `fan_out`, the algorithm learns a matrix `U` of size `fan_in` $\times$ `rank`, a matrix `V` of size `rank` $\times$ `fan_out`, and a matrix of size `number_of_scenes` $\times$ `rank`, where `rank` denotes the dimension of the matrix space.
