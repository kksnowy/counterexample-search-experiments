--------------------------------------------------------------------------------
-- Inputs and outputs

-- Define the type for our input images
type Image = Tensor Real [28, 28]

-- The type of the output labels
-- i.e a number between 0 and 9, one for each digit
type Label = Index 10

-- A predicate that states that all the pixel values in a given image are
-- in the range 0.0 to 1.0
validImage : Image -> Bool
validImage x = forall i j . 0 <= x ! i ! j <= 1

--------------------------------------------------------------------------------
-- Network

-- Declare the network used to classify images. The output of the network is a
-- score for each of the digits 0 to 9.
@network
classifier : Image -> Tensor Real [10]

-- The classifier advises that input image `x` has label `i` if the score
-- for label `i` is greater than the score of any other label `j`.
advises : Image -> Label -> Bool
advises x i = forall j . j != i => classifier x ! i > classifier x ! j

--------------------------------------------------------------------------------
-- Definition of robustness around a point

-- First we define the parameter `epsilon` that will represent the radius of the
-- ball that we want the network to be robust in. Note that we declare this as
-- a parameter which allows the value of `epsilon` to be specified at compile
-- time rather than be fixed in the specification.
@parameter
epsilon : Real

-- Next we define what it means for an image `x` to be in a ball of
-- size epsilon around 0.
boundedByEpsilon : Image -> Bool
boundedByEpsilon x = forall i j . -epsilon <= x ! i ! j <= epsilon

-- We now define what it means for the network to be robust around an image `x`
-- that should be classified as `y`. Namely, that for any perturbation no greater
-- than epsilon then if the perturbed image is still a valid image then the
-- network should still advise label `y` for the perturbed version of `x`.
robustAround : Image -> Label -> Bool
robustAround image label = forall perturbation .
  let perturbedImage = image - perturbation in
  boundedByEpsilon perturbation and validImage perturbedImage =>
    advises perturbedImage label

--------------------------------------------------------------------------------
-- Robustness around the first image and label in a set of two images and labels
-- Adapted from the mnist-robustness.vcl specification available in the 
-- examples/mnist-robustness directory within the Vehicle codebase

@dataset
trainingImages : Vector Image 2

@dataset
trainingLabels : Vector Label 2

@property
robustSingleImage : Bool
robustSingleImage = robustAround (trainingImages ! 0) (trainingLabels ! 0)
