package com.github.alleveenstra.neuralwordembedding.tools;

import org.jblas.DoubleMatrix;

import java.util.ArrayList;

public class SupervisedExampleSet {

    public static SupervisedExampleSet XOR, SIN, TANH, RANDOM;

    static {
        XOR = xor();
        SIN = sin();
        TANH = tanh();
        RANDOM = random();
    }

    public DoubleMatrix examples[];
    public DoubleMatrix classes[];

    public static SupervisedExampleSet xor() {
        final SupervisedExampleSet supervisedExampleSet = new SupervisedExampleSet();
        supervisedExampleSet.examples = new DoubleMatrix[4];
        supervisedExampleSet.classes = new DoubleMatrix[4];

        // 1 && 1 -> -1
        supervisedExampleSet.examples[0] = DoubleMatrix.ones(1, 2);
        supervisedExampleSet.classes[0] = DoubleMatrix.zeros(1).put(0, -1);

        // -1 && 1 -> 1
        supervisedExampleSet.examples[1] = DoubleMatrix.ones(1, 2).put(0, -1);
        supervisedExampleSet.classes[1] = DoubleMatrix.ones(1);

        // 1 && -1 -> 1
        supervisedExampleSet.examples[2] = DoubleMatrix.ones(1, 2).put(1, -1);
        supervisedExampleSet.classes[2] = DoubleMatrix.ones(1);

        // -1 && -1 -> -1
        supervisedExampleSet.examples[3] = DoubleMatrix.zeros(1, 2).put(0, -1).put(1, -1);
        supervisedExampleSet.classes[3] = DoubleMatrix.zeros(1).put(0, -1);

        return supervisedExampleSet;
    }

    public static SupervisedExampleSet sin() {
        final ArrayList<DoubleMatrix> examples = new ArrayList<>();
        final ArrayList<DoubleMatrix> classes = new ArrayList<>();
        for (double v = -Math.PI; v < Math.PI; v+= 0.1) {
            examples.add(DoubleMatrix.zeros(1, 1).add(v));
            classes.add(DoubleMatrix.zeros(1, 1).add(Math.sin(v)));
        }
        SupervisedExampleSet supervisedExampleSet = new SupervisedExampleSet();
        supervisedExampleSet.examples = new DoubleMatrix[examples.size()];
        supervisedExampleSet.classes = new DoubleMatrix[classes.size()];
        for (int i = 0; i < examples.size(); i++) {
            supervisedExampleSet.examples[i] = examples.get(i);
            supervisedExampleSet.classes[i] = classes.get(i);
        }
        return supervisedExampleSet;
    }

    public static SupervisedExampleSet tanh() {
        final ArrayList<DoubleMatrix> examples = new ArrayList<>();
        final ArrayList<DoubleMatrix> classes = new ArrayList<>();
        for (double v = -Math.PI; v < Math.PI; v+= 0.1) {
            examples.add(DoubleMatrix.zeros(1, 1).add(v));
            classes.add(DoubleMatrix.zeros(1, 1).add(Math.tanh(v)));
        }
        SupervisedExampleSet supervisedExampleSet = new SupervisedExampleSet();
        supervisedExampleSet.examples = new DoubleMatrix[examples.size()];
        supervisedExampleSet.classes = new DoubleMatrix[classes.size()];
        for (int i = 0; i < examples.size(); i++) {
            supervisedExampleSet.examples[i] = examples.get(i);
            supervisedExampleSet.classes[i] = classes.get(i);
        }
        return supervisedExampleSet;
    }

    public static SupervisedExampleSet random() {
        final SupervisedExampleSet supervisedExampleSet = new SupervisedExampleSet();
        supervisedExampleSet.examples = new DoubleMatrix[4];
        supervisedExampleSet.classes = new DoubleMatrix[4];

        supervisedExampleSet.examples[0] = DoubleMatrix.randn(1, 2).min(0.5);
        supervisedExampleSet.classes[0] = DoubleMatrix.randn(1);

        supervisedExampleSet.examples[1] = DoubleMatrix.randn(1, 2).min(0.5);
        supervisedExampleSet.classes[1] = DoubleMatrix.randn(1);

        supervisedExampleSet.examples[2] = DoubleMatrix.randn(1, 2).min(0.5);
        supervisedExampleSet.classes[2] = DoubleMatrix.randn(1);

        supervisedExampleSet.examples[3] = DoubleMatrix.randn(1, 2).min(0.5);
        supervisedExampleSet.classes[3] = DoubleMatrix.randn(1);

        return supervisedExampleSet;
    }
}