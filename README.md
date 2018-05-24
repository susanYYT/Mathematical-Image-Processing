# Mathematical-Image-Processing
International Workshop on Mathematical Image Processing: Models, Theory and Algorithms
  
  “数字图像处理的模型、理论和算法”国际研讨会
  
  2018年4月13-15日
  
  南京邮电大学

## 报告题目和摘要
## Michael Ng (Hong Kong Baptist University)
### Multi-Label Classification by Semi-Supervised Singular Value Decomposition
Multi-label problems arise in various domains, including automatic multimedia data
categorization, and have generated significant interest in computer vision and machine
learning community. In this talk, we proposed to use a semi-supervised singular value
decomposition (SVD) to effectively capture the label correlations.
Experimental results for synthetic and real-world multimedia data sets demonstrate
that the proposed method can exploit the label correlations and obtain promising and
better label prediction results than the state-of-the-art methods.
## 雷皓（中国科学院武汉物理与数学研究所）
### 磁共振成像图像处理中的数学问题
磁共振成像的技术瓶颈问题主要有两个：1）如何快速、廉价地获取图像，以降
低MRI检查成本，使得高通量检查或筛查成为可能；2）如何从图像中提取定量、个
体化的信息用于指导临床诊疗。这两个问题的最终解决都离不开数学。例如，稀疏
采样平行快速成像是近年来发展起来的、并已得到广泛应用的快速成像方法，而该
方法的实现在很大程度上依赖于压缩感知理论的提出和完善。著名数学家陶哲轩等
在这方面起到了决定性的作用。磁共振图像信息的提取目前主要依靠两种方法：放
射专家读片和基于统计方法的参数分析。前者可认为是个性化的信息提取，但在定
量性、客观性、可重复性等方面可能存在不足；后者是定量、客观的分析，但尚不
能高效实现个体化信息的提取。第二个瓶颈问题的解决需要数学不同分支的广泛参
与，包括偏微分方程、流体几何、图论、模式识别、网络分析、模型选择、数理统
计、复杂机器学习等。
## 董彬（北京大学）
### “Deep Revolution” in Image Restoration and Beyond
Deep learning continues to dominate machine learning. It is now widely used in
many research areas in science and engineering, and has major industrial impacts. Deep
learning methods have achieved remarkable results in a variety of tasks, especially in a
supervised learning environment. They have surpassed, or as good as, human in Go,
playing video games, accurately identifying objects in images and videos, diagnosing
certain diseases from medical images, etc.
In this talk, I will start with a brief review of classical (pre deep learning) image
restoration methods, followed by some recent applications of deep learning in image
restoration and image analysis. I will present my personal understanding of deep learning
in image restoration from the perspective of applied mathematics, which inspired two of
our recent work. One work is on combining numerical differential equation and deep
convolutional architecture design. In this work, we interpret some of the state-of-the-art
deep CNNs, such as ResNet, FractalNet, PolyNet, RevNet, etc., in terms of numerical
(stochastic) differential equations; and to propose new deep architectures that can further
improve the prediction accuracy of the existing networks in image classification. In the
other work, we proposed an end-to-end model for imaging based diagnosis in medical
imaging. Unlike traditional methods where image reconstruction and image recognition
are treated as two separate steps, our end-to-end model merges the two steps into one.
Our numerical experiments on a large scale real CT data set demonstrated the benefit of
the proposed method.
## 曾铁勇（香港中文大学）
### Convex and Non-Convex Optimization in Image Recovery and Segmentation
We will report some progress on the convex and non-convex approaches for image restoration
and segmentation.
## 吴春林（南开大学）
### A General Truncated Regularization Framework for Contrast-Preserving
Variational Signal and Image Restoration: Motivation and Implementation
Variational methods have become an important kind of methods in signal and image
restoration - a typical inverse problem. One important minimization model consists of the
squared L2 data fidelity (corresponding to Gaussian noise) and a regularization term
constructed by a potential function composed of first order difference operators. It is well
known that total variation (TV) regularization, although achieved great successes, suffers
from a contrast reduction effect. Using a typical signal, we show that, actually all convex
regularizers and most nonconvex regularizers have this effect. With this motivation, we
present a general truncated regularization framework. The potential function is a
truncation of existing nonsmooth potential functions and thus flat from some positive t.
Some analysis in 1D theoretically demonstrate the good contrast-preserving ability of the
framework. We also give optimization algorithms with convergence verification in 2D,
where global minimizers of each subproblem (either convex or nonconvenx) are
calculated. Experiments numerically show the advantages of the framework.
## 崔宰珪（上海交通大学）
### PET-MRI Joint Reconstruction by Joint Sparsity Based Tight Frame Regularization
Recent technical advances lead to the coupling of PET and MRI scanners, enabling
to acquire functional and anatomical data simultaneously. In this talk, we propose a tight
frame based PET-MRI joint reconstruction model via the joint sparsity of tight frame
coefficients. In addition, a non-convex balanced approach is adopted to take the different
regularities of PET and MRI images into account. To solve the nonconvex and
nonsmooth model, a proximal alternating minimization algorithm is proposed, and the
global convergence is present based on Kurdyka-Lojasiewicz property. Finally, the
numerical experiments show that our proposed models achieve better performance over
the existing PET-MRI joint reconstruction models.
## 常谦顺（江苏师范大学）
### An Adaptive Algorithm for TV-based Model of Three Norms in Image Restoration .
## 段玉萍（天津大学）
### Accurate MR reconstruction with correction for intensity inhomogeneity
High-field Magnetic Resonance Imaging (MRI) becomes popular due to the benefits
of potential higher signal-to-noise ratio, contrast-to-noise ratios and spectral resolution, a
byproduct of which is the intensity inhomogeneity (i.e., bias field). In this work, we
develop a novel MRI reconstruction method by regarding the reconstructed image as a
combination of the true intensity and a bias filed. The undersampled MRI
reconstruction is formulated as a least-square problem integrating with an inf-convolution
of the first-order and second-order regularisers, and the shearlet transform. More
specifically, we use the total variation and total variation of the gradient to guarantee the
properties of the true intensity (piecewise constant) and bias field (spatially smooth). The
shearlet transform is employed to capture the anisotropic features such as edges, curves,
and so on. The proposed model is solved by splitting variables and Alternating Direction
Method of Multipliers (ADMM), where all subproblems have the closed-form solutions.
Numerical experiments on both phantom and MRI data are conducted to demonstrate the
advantageous of the proposed method in reconstruction of high-field MRI.
## 沈超敏（华东师范大学）
### Global Nonlinear Metric Learning by Gluing Local Linear Metrics
We address the nonlinear metric learning by constructing a smooth nonlinear metric
from the data. First, we locally define an initial linear metric on each cluster by principal
component analysis. Second, we glue such local linear metrics to form a smooth
nonlinear metric by a partition of unity on the sample space, and further learn the global
nonlinear metric. Third, we conduct the intrinsic steepest descent algorithm on matrix
manifolds for implementation. Finally, we compare our approach with several
state-of-the-art methods on a variety of datasets. The results validate that the robustness
and accuracy of classification are both improved under our nonlinear metric. The novelty
of our global smooth nonlinear metric learning model lies in that it has completely
overcome drawbacks of local metric learning methods: the partition coefficients obtained
by the partition of unity is smooth, while the metric at any point on the manifold can be
directly defined.
## 刘君（北京师范大学）
### Normalized Cut with Adaptive Similarity and Spatial Regularization
In this talk, we propose a normalized cut segmentation algorithm with spatial
regularization priority and adaptive similarity matrix. We integrate the well-known
expectation-maximum(EM) method in statistics and the regularization technique in
partial differential equation (PDE) method into the normalized cut. The introduced EM
technique makes our method can adaptively update the similarity matrix. This step can be
regarded as we build a simple generator to produce some better similarity matrices for
classification criterion.
While the regularization priori can guarantee that the proposed algorithm uses a
spatially regularized spectrum vector as discriminator to classify pixels. The generator
and discriminator cooperate with each other and makes the proposed algorithm has a
robust performance under noise.
To unify the three totally different methods including EM, spatial regularization, and
spectral graph clustering, we built a variational framework to combine them and get a
general normalized cut segmentation algorithm. The well-defined theory of the proposed
model is also given in the paper.
Compared with some existing spectral clustering methods such as the traditional
normalized cut algorithm and the variational based Chan-Vese model, numerical
experiments show that our methods can achieve promising segmentation performance.
## 沈纯理（华东师范大学）
### 曲面点云图像的去噪、去模糊问题及其在脑电波（EEG）重建中的应用
1. 曲面点云图像的去噪、去模糊处理的关键是对以连续方式或以离散点集方式
表达的曲面上图像，如何去表达它的全变分(Total Variation).
  
  2. 利用微分几何的方法对离散化的曲面S 求出了曲面上各点的法向及主方
向，并给出了曲面S 上图像u 的全变分TV(u) 的具体表达式，从而图像u 的去
噪、去模糊问题就可归纳为常规的计算能量泛函的极小值问题。
  
  3. 利用Chambolle-Pock 方法及随机梯度算法快速求解泛函的极小值问题。
  
  4. 我们将此方法应用于脑电波的重建问题，即根据脑电图电极的电位测量值
(measurable potentials at EEG electrodes)反推出脑电波的源值(EEG electrical source).
## 庞志峰（河南大学）
### Half-quadratic adaptive $TV^p$ to the image restoration problem
To keep structures in the restoration problem is very important via coupling the local
information of the image with the proposed model. In this paper we propose a local
self-adaptive $\ell^p$-regularization model for $p\in(0,2)$ based on the total variation
scheme, where the choice of $p$ depends on the local structures described by the
eigenvalues of the structure tensor. Since the proposed model as the classic
$\ell^p$ problem unifies two classes of optimization problems such as the nonconvex and
nonsmooth problem when $p\in(0,1)$, and the convex and smooth problem when
$p\in(1,2)$, it is generally challenging to find a ready algorithmic framework to solve it.
Here we propose a new and robust numerical method via coupling with the half-quadratic
scheme and the alternating direction method of multipliers(ADMM). The convergence of
the proposed algorithm is established and the numerical experiments illustrate the
possible advantages of the proposed model and numerical methods over some existing
variational-based models and methods.
## 袁强强（武汉大学）
### 深度学习在遥感信息质量改善中的应用
遥感影像质量改善是遥感信息处理研究中的热门问题。针对传统的正则化模型
方法存在参数依赖，普适性不足的缺点，本报告主要介绍本团队利用深度学习技术
在遥感影像去噪、修复、多源信息融合以及定量反演方面所做的一些新的尝试。
## 张雄军（华中师范大学）
### A Fast Algorithm for Deconvolution and Poisson Noise Removal
Poisson noise removal problems have attracted much attention in recent years. The
main aim of this paper is to study and propose an alternating minimization algorithm for
Poisson noise removal with nonnegative constraint. The algorithm minimizes the sum of
a Kullback-Leibler divergence term and a total variation term. We derive the algorithm by
utilizing the quadratic penalty function technique. Moreover, the convergence of the
proposed algorithm is also established under very mild conditions. Numerical
comparisons between our approach and several state-of-the-art algorithms are presented
to demonstrate the efficiency of our proposed algorithm. (This is a joint work with
Michael K. Ng and Minru Bai).
## 姜明（北京大学）
### Asynchronous Parallel Computation for Image Reconstruction
With the rapid advance of computer hardware such as multi-core CPU/GPU,
multi-node supercomputer, FPGA, asynchronous parallel computation becomes necessary
for making the best use of such computing devices. In this talk, asynchronous parallel
computation will be discussed from the following perspectives, including
energy-efficiency, communication model, and architecture and implementation for
iterative algorithms for image reconstruction. The implementation of the Mumford-shah
regularization for x-ray CT and electron tomography is used as a demonstration for
implementation.
## 文有为（湖南师范大学）
### A Fast Proximal Gradient Algorithm For Single Particle Reconstruction of Cryo-EM
We consider the problem of single particle reconstruction (SPR) from cryo-electron
microscopy (cryo-EM), where the three-dimensional (3D) structure of particle is
reconstructed from the many noisy two-dimension (2D) projected and blurred images. In
this talking, single particle reconstruction is represented by solving an linear inverse
problem with perturbations. Regularization method is applied to solve the linear system
since it is ill-posed. We apply Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)
to find the solution of the $L_1$ regularized optimization problem. Numerical
experiments with simulated images demonstrate that the proposed methods significantly
reduce the estimation error and improved reconstruction quality.
## 应时辉（上海大学）
### 基于流形分布的图像处理与分析
该报告针对基于影像组学的图像处理与分析方法进行介绍。特别是图像配准与
标准化问题、纵向图像演化过程刻画。具体地，首先介绍两幅图像配准的数学模型
与算法；其次，通过将图像在图像流形上的分布信息引入影像组标准化模型，得到
无偏图谱建立方法；再次，通过图像流形上的路径回归方法，得到婴幼儿大脑发育
过程的影像演化过程。
## 黄玉梅（兰州大学）
### Rank Minimization with Applications to Image Noise Removal
Rank minimization problem has a wide range of applications in different areas.
However, since this problem is NP-hard and non-convex, the frequently used method is
to replace the matrix rank minimization with nuclear norm minimization. Nuclear norm is
the convex envelope of the matrix rank and it is more computationally tractable. Matrix
completion is a special case of rank minimization problem. In this talk, we consider
directly using matrix rank as the regularization term instead of nuclear norm in the cost
function for matrix completion problem. The solution is analyzed and obtained by a
hard-thresholding operation on the singular values of the observed matrix. Then by
exploiting patch-based nonlocal self-similarity scheme, we apply the proposed rank
minimization algorithm to remove white Gaussian additive noise in images. Gamma
multiplicative noise is also removed in logarithm domain. The experimental results
illustrate that the proposed algorithm can remove noises in images more efficiently than
nuclear norm can do. And the results are also competitive with those obtained by using
the existing state-of-the-art noise removal methods in the literature.
## 黎芳（华东师范大学）
### Image restoration with patch-based low rank regularization
In this talk, we propose new decoupled variational models for image restoration
based on patch-based low rank regularization with nuclear norm minimization. Some
mathematical analysis of the models and the algorithms are given. The numerical
experiments and comparisons on various images demonstrate the effectiveness of the
proposed methods.
## 吕良福（天津大学）
### Weighted nuclear norm minimization for tensor completion using tensor-SVD
Tensor nuclear norm minimization as the extension of nuclear norm minimization in
the field of tensor domain, has attracted extensive attention in the fields of computer
vision and neuroscience. However, in order to obtain higher accuracy in practical
application, many researchers prefer to using weighted tensor nuclear norm minimization
rather than tensor nuclear norm minimization. Furthermore, a new tensor
decomposition,which is called tensor-SVD, is proposed for utilizing the relationship
among slices of tensor. In this paper, we propose the weighted tensor nuclear norm
minimization to approximate tensor completion problem under the framework of
tensor-SVD. Then we use alternating direction method of multipliers method solve it,
verify the its convergence, and proof its limit point satisfying the KKT condition.
Furthermore, our proposed method shows a significant improvement with respect to the
accuracy in comparison with tensor nuclear norm minimization, and achieves
state-of-the-art performance in typical low level vision tasks, including video completion,
image inpainting et al.
## 方发明（华东师范大学）
### Sparse Unmixing of Hyperspectral Images
Spectral unmixing aims at estimating the proportions (abundances) of pure
spectrums (endmembers) in each mixed pixel of hyperspectral data. Recently, the
semi-supervised approach, which takes the spectral library as prior knowledge, has been
attracting much attention in unmixing. In this talk, we will present two new
semi-supervised unmixing models. Firstly, we show a novel unmixing model combined
with two effective regularization terms: a similarity-weighting constraint and the Lp (0 <
p < 1) norm sparse regularization. Secondly, a framelet-based sparse unmixing model is
presented. This model can promote the abundance sparsity in framelet domain and
discriminates the approximation and detail components of hyperspectral data after
framelet decomposition. In both of the models, the iteration based algorithms are
discussed to obtain the minimal solution. Experimental results on simulated and real data
demonstrate that our models are promising.
