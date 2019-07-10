import FastaiNotebook_09_optimizer
import Path
import TensorFlow

let path = downloadImagenette()
let il = ItemList(fromFolder: path, extensions: ["jpeg", "jpg"])
let sd = SplitData(il, fromFunc: {grandParentSplitter(fName: $0, valid: "val")})
var procLabel = CategoryProcessor()
let sld = makeLabeledData(sd, fromFunc: parentLabeler, procLabel: &procLabel)
let rawData = sld.toDataBunch(itemToTensor: pathsToTensor, labelToTensor: intsToTensor)
let data = transformData(rawData, tfmItem: { openAndResize(fname: $0, size: 128) })


// TODO(TF-619): Remove this, and use CNNModel from 08 instead.

public struct ThisModuleFABatchNorm<Scalar: TensorFlowFloatingPoint>: FALayer {
  // Configuration hyperparameters
  @noDerivative var momentum, epsilon: Tensor<Scalar>
  // Running statistics
  @noDerivative let runningMean, runningVariance: Reference<Tensor<Scalar>>
  // Trainable parameters
  public var scale, offset: Tensor<Scalar>

  public init(featureCount: Int, momentum: Scalar, epsilon: Scalar = 1e-5) {
    self.momentum = Tensor(momentum)
    self.epsilon = Tensor(epsilon)
    self.scale = Tensor(ones: [featureCount])
    self.offset = Tensor(zeros: [featureCount])
    self.runningMean = Reference(Tensor(0))
    self.runningVariance = Reference(Tensor(1))
  }

  public init(featureCount: Int, epsilon: Scalar = 1e-5) {
    self.init(featureCount: featureCount, momentum: 0.9, epsilon: epsilon)
  }

  @differentiable
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    let mean = input.mean(alongAxes: [0, 1, 2])
    let variance = input.variance(alongAxes: [0, 1, 2])
    runningMean.value += (mean - runningMean.value) * (1 - momentum)
    runningVariance.value += (variance - runningVariance.value) * (1 - momentum)
    let normalizer = rsqrt(variance + epsilon) * scale
    return (input - mean) * normalizer + offset
  }

  @differentiable
  public func forwardInference(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    let mean = runningMean.value
    let variance = runningVariance.value
    let normalizer = rsqrt(variance + epsilon) * scale
    return (input - mean) * normalizer + offset
  }
}

public struct ThisModuleConvBN<Scalar: TensorFlowFloatingPoint>: FALayer {
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>
  public var conv: FANoBiasConv2D<Scalar>
  public var norm: ThisModuleFABatchNorm<Scalar>

  public init(_ cIn: Int, _ cOut: Int, ks: Int = 3, stride: Int = 1){
    // TODO (when control flow AD works): use Conv2D without bias
    self.conv = FANoBiasConv2D(cIn, cOut, ks: ks, stride: stride, activation: relu)
    self.norm = ThisModuleFABatchNorm(featureCount: cOut, epsilon: 1e-5)
  }

  @differentiable
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    return norm(conv(input))
  }
}

public struct CNNModel: Layer {
  public var convs: [ThisModuleConvBN<Float>]
  public var pool = FAGlobalAvgPool2D<Float>()
  public var linear: FADense<Float>

  public init(channelIn: Int, nOut: Int, filters: [Int]){
    convs = []
    let (l1,l2) = (channelIn, prevPow2(channelIn * 9))
    convs = [ThisModuleConvBN(l1,   l2,   stride: 1),
             ThisModuleConvBN(l2,   l2*2, stride: 2),
             ThisModuleConvBN(l2*2, l2*4, stride: 2)]
    let allFilters = [l2*4] + filters
    for i in 0..<filters.count { convs.append(ThisModuleConvBN(allFilters[i], allFilters[i+1], stride: 2)) }
    linear = FADense<Float>(filters.last!, nOut)
  }

  @differentiable
  public func callAsFunction(_ input: TF) -> TF {
    // TODO: Work around https://bugs.swift.org/browse/TF-606
    return linear(pool(convs(input)))
  }
}

func modelInit() -> CNNModel { return CNNModel(channelIn: 3, nOut: 10, filters: [64, 64, 128, 256]) }

var hps: [String:Float] = [HyperParams.lr: 0.01]
func optFunc(_ model: CNNModel) -> StatefulOptimizer<CNNModel> {
  return StatefulOptimizer(for: model, steppers: [SGDStep()], stats: [], hps: hps)
}
var learner = Learner(data: data, lossFunc: softmaxCrossEntropy, optFunc: optFunc, modelInit: modelInit)
var recorder = learner.makeDefaultDelegates(metrics: [accuracy])
learner.delegates.append(learner.makeNormalize(mean: mnistStats.mean, std: mnistStats.std))
try! learner.fit(1)

