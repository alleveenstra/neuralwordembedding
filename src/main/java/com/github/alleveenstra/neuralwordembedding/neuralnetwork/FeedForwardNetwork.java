package com.github.alleveenstra.neuralwordembedding.neuralnetwork;

import org.jblas.DoubleMatrix;

import java.io.Serializable;

public class FeedForwardNetwork implements Serializable {
    private static final long serialVersionUID = -9008979005651977847L;

    protected boolean extraInput;
    protected int inputSize;
    protected int outputSize;
    protected int hiddenSize;
    protected int[] hiddenShape;
    protected int nLayers;

    protected int[] size;
    protected DoubleMatrix[] activations;
    protected DoubleMatrix[] biases;
    protected DoubleMatrix[] previousUpdate;
    protected DoubleMatrix[] weights;
    protected DoubleMatrix[] deltas;

    private double beta;

    public FeedForwardNetwork(int inputSize, int... shape) {
        this(false, inputSize, shape);
    }

    public FeedForwardNetwork(boolean extraInput, int inputSize, int... shape) {
        this.extraInput = extraInput;
        if (this.extraInput) {
            inputSize++;
        }
        this.inputSize = inputSize;
        this.outputSize = shape[shape.length - 1];
        this.hiddenSize = shape.length;
        this.hiddenShape = shape;

        this.nLayers = 1 + this.hiddenSize;

        this.size = new int[nLayers];
        this.activations = new DoubleMatrix[nLayers];
        this.biases = new DoubleMatrix[nLayers];
        this.previousUpdate = new DoubleMatrix[nLayers];
        this.weights = new DoubleMatrix[nLayers];
        this.deltas = new DoubleMatrix[nLayers];

        this.size[0] = this.inputSize;
        this.activations[0] = DoubleMatrix.ones(1, this.inputSize);

        int hiddenNeurons = 0;
        for (int layer = 0; layer < shape.length - 1; layer++) {
            hiddenNeurons += shape[layer];
        }
        this.beta = 0.7 * Math.pow(hiddenNeurons, 1.0 / this.inputSize);

        int prevLayerSize;
        for (int layer = 0; layer < this.hiddenSize; layer++) {
            if (layer == 0) {
                prevLayerSize = this.inputSize;
            } else {
                prevLayerSize = this.hiddenShape[layer - 1];
            }
            int layerSize = this.hiddenShape[layer];
            this.size[layer + 1] = layerSize;
            this.previousUpdate[layer + 1] = DoubleMatrix.zeros(prevLayerSize, layerSize);
            this.weights[layer + 1] = rand(DoubleMatrix.zeros(prevLayerSize, layerSize), layer + 1);
            this.deltas[layer + 1] = DoubleMatrix.zeros(1, layerSize);
            this.activations[layer + 1] = DoubleMatrix.ones(1, layerSize);
            this.biases[layer + 1] = DoubleMatrix.zeros(1, layerSize);
        }
    }

    private DoubleMatrix rand(DoubleMatrix matrix, int layer) {
        for (int index = 0; index < matrix.length; ++index) {
            matrix.put(index, (Math.random() - 0.5) * 2);
        }
        int nOut = this.size[layer - 1];
        int nIn = this.size[layer];
        for (int in = 0; in < nIn; in++) {
            double n = 0.0;
            for (int out = 0; out < nOut; out++) {
                double w = matrix.get(out, in);
                n += w * w;
            }
            n = Math.sqrt(n);
            for (int out = 0; out < nOut; out++) {
                double w = matrix.get(out, in);
                w = beta * w / n;
                matrix.put(out, in, w);
            }
        }
        return matrix;
    }
}
