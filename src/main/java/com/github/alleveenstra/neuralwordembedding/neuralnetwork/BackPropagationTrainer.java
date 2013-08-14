package com.github.alleveenstra.neuralwordembedding.neuralnetwork;

import org.jblas.DoubleMatrix;

public class BackPropagationTrainer {
    public double eta = 0.25, eta_bias = 0.0, eta_L1 = 0.0, eta_L2 = 0.0, eta_momentum = 0.0, eta_decay = 0.0;
    private FeedForwardNetwork network;


    public BackPropagationTrainer(FeedForwardNetwork network) {
        this.network = network;
    }

    public DoubleMatrix evaluate(DoubleMatrix inputs) {
        if ((network.extraInput && inputs.length + 1 != network.inputSize) ||
                !network.extraInput && inputs.length != network.inputSize){
            throw new IllegalArgumentException("Input vector of incorrect size");
        }

        // copy the input into the input network activation
        for (int input = 0; input < inputs.length; input++) {
            network.activations[0].put(input, inputs.get(input));
        }

        this.feedForward();

        return network.activations[network.nLayers - 1];
    }

    public void feedForward() {
        for (int layer = 1; layer < network.nLayers; layer++) {
            // network.activations[layer] = this.activation(network.activations[layer - 1] * network.weights[layer] + network.biases[layer]);
            final DoubleMatrix incomingActivation = network.activations[layer - 1].mmul(network.weights[layer]);
            if (eta_bias > 0.0) {
                incomingActivation.addi(network.biases[layer]);
            }
            activation(incomingActivation);
            network.activations[layer] = incomingActivation;
        }
    }

    private String getDims(DoubleMatrix matrix) {
        return matrix.getRows() + "x" +matrix.getColumns();
    }

    public DoubleMatrix[] calculateDeltas(DoubleMatrix target) {
        DoubleMatrix[] result = new DoubleMatrix[network.nLayers];
        // for layer in reversed(range(1, network.nLayers)):
        for (int layer = network.nLayers - 1; layer >= 1; layer--) {
            if (layer == network.nLayers - 1) {
                // network.deltas[layer] = numpy.multiply(self.derivative(network.activations[layer]),
                //                                       (target - network.activations[layer]))
                DoubleMatrix error = target.sub(network.activations[layer]);
                DoubleMatrix deriv = network.activations[layer].dup();
                derivation(deriv);
                error.muli(deriv, network.deltas[layer]);
            } else {
                // network.deltas[layer] = numpy.multiply(self.derivative(network.activations[layer]),
                //                                       (network.deltas[layer + 1] * network.weights[layer + 1].T))
                //logger.info(layer + " " + getDims(network.deltas[layer + 1]) + " " + getDims(network.weights[layer + 1].transpose()));
                DoubleMatrix error = network.deltas[layer + 1].mmul(network.weights[layer + 1].transpose());
                DoubleMatrix deriv = network.activations[layer].dup();
                derivation(deriv);
                error.muli(deriv, network.deltas[layer]);
            }
            // network.biases[layer] += network.deltas[layer] * this.eta_bias
            result[layer] = network.deltas[layer].mul(this.eta_bias);
        }
        return result;
    }

    public DoubleMatrix[] calculateWeightUpdate() {
        DoubleMatrix[] weightUpdate = new DoubleMatrix[network.nLayers];
        for (int layer = 1; layer < network.nLayers; ++layer) {
            // copy delta only if regularization is enabled
            DoubleMatrix delta = this.eta_L1 > 0 || this.eta_L2 > 0 ? network.deltas[layer].dup() : network.deltas[layer];

            // L1 = numpy.signi(network.activations[layer]) * -this.eta_L1
            if (this.eta_L1 > 0.0) {
                DoubleMatrix L1 = signi(network.activations[layer].dup()).muli(-this.eta_L1);
                delta.addi(L1);
            }

            // L2 = network.activations[layer] * -this.eta_L2
            if (this.eta_L2 > 0.0) {
                DoubleMatrix L2 = network.activations[layer].mul(-this.eta_L2);
                delta.addi(L2);
            }

            // update = (network.activations[layer - 1].T * (network.deltas[layer] + L1 + L2))
            final DoubleMatrix activationT = network.activations[layer - 1].transpose();
            weightUpdate[layer] = activationT.mmul(delta);

            // update += network.previousUpdate[layer] * this.eta_momentum
            if (eta_momentum > 0.0) {
                network.previousUpdate[layer].muli(this.eta_momentum);
                weightUpdate[layer].addi(network.previousUpdate[layer]);
            }

            // update -= network.weights[layer] * this.eta_decay
            if (eta_decay > 0.0) {
                final DoubleMatrix decay = network.weights[layer].mul(eta_decay);
                weightUpdate[layer].subi(decay);
            }

            // * eta
            weightUpdate[layer].muli(eta);

            // network.previousUpdate[layer] = update
            network.previousUpdate[layer] = weightUpdate[layer];
        }
        return weightUpdate;
    }

    public void applyWeightUpdate(DoubleMatrix[] biasUpdate, DoubleMatrix[] weightUpdate) {
        if (weightUpdate != null) {
            assert weightUpdate.length == network.nLayers;
            for (int layer = 1; layer < network.nLayers; ++layer) {
                network.weights[layer].addi(weightUpdate[layer]);
            }
        }
        if (biasUpdate != null) {
            assert biasUpdate.length == network.nLayers;
            for (int layer = 1; layer < network.nLayers; ++layer) {
                network.biases[layer].addi(biasUpdate[layer]);
            }
        }
    }

    private DoubleMatrix signi(DoubleMatrix matrix) {
        for (int index = 0; index < matrix.length; ++index) {
            matrix.put(index, Math.signum(matrix.get(index)));
        }
        return matrix;
    }

    public void activation(DoubleMatrix matrix) {
        for (int index = 0; index < matrix.length; index++) {
            matrix.put(index, activationFunction(matrix.get(index)));
        }
    }

    private double activationFunction(double value) {
        return Math.tanh(value);
    }

    public void derivation(DoubleMatrix matrix) {
        for (int index = 0; index < matrix.length; index++) {
            matrix.put(index, derivedActivationFunction(matrix.get(index)));
        }
    }

    public double derivedActivationFunction(double value) {
        return 1.0 - value * value;
    }
}
