package com.github.alleveenstra.neuralwordembedding.tools;

import com.github.alleveenstra.neuralwordembedding.neuralnetwork.BackPropagationTrainer;
import com.github.alleveenstra.neuralwordembedding.neuralnetwork.FeedForwardNetwork;
import com.github.alleveenstra.neuralwordembedding.neuralnetwork.ZealousWordEmbeddingTrainingStrategy;
import com.google.common.collect.Lists;
import org.jblas.DoubleMatrix;
import org.junit.Test;

import java.util.*;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.lessThan;

public class TestNN {
    @Test
    public void verify_gradient() {
        final double meanDifference = meanGradientDifference();
        assertThat(meanDifference, lessThan(0.0001));
    }

    private static double meanGradientDifference() {
        FeedForwardNetwork ffn = new FeedForwardNetwork(2, 3, 2, 1);
        BackPropagationTrainer bpTrain = new BackPropagationTrainer(ffn);
        bpTrain.eta = 0.1;
        bpTrain.eta_bias = 0.05;
        bpTrain.eta_L1 = 0.1;
        bpTrain.eta_L2 = 0.1;
        bpTrain.eta_momentum = 0.1;
        Map<String, DoubleMatrix> embeddings = new HashMap<>();
        embeddings.put("the", DoubleMatrix.rand(50));
        embeddings.put("quick", DoubleMatrix.rand(50));
        embeddings.put("brown", DoubleMatrix.rand(50));
        embeddings.put("fox", DoubleMatrix.rand(50));
        embeddings.put("jumps", DoubleMatrix.rand(50));
        embeddings.put("over", DoubleMatrix.rand(50));
        embeddings.put("lazy", DoubleMatrix.rand(50));
        embeddings.put("dog", DoubleMatrix.rand(50));
        List<Vector<String>> sentences = new ArrayList<>();
        sentences.add(new Vector<>(Lists.newArrayList("the", "quick", "brown", "vox", "jumps", "over", "the", "lazy", "dog", "and")));
        sentences.add(new Vector<>(Lists.newArrayList("the", "quick", "brown", "vox", "jumps", "over", "the", "lazy", "dog", "and")));
        sentences.add(new Vector<>(Lists.newArrayList("the", "quick", "brown", "vox", "jumps", "over", "the", "lazy", "dog", "and")));
        sentences.add(new Vector<>(Lists.newArrayList("the", "quick", "brown", "vox", "jumps", "over", "the", "lazy", "dog", "and")));
        sentences.add(new Vector<>(Lists.newArrayList("the", "quick", "brown", "vox", "jumps", "over", "the", "lazy", "dog", "and")));
        sentences.add(new Vector<>(Lists.newArrayList("the", "quick", "brown", "vox", "jumps", "over", "the", "lazy", "dog", "and")));
        sentences.add(new Vector<>(Lists.newArrayList("the", "quick", "brown", "vox", "jumps", "over", "the", "lazy", "dog", "and")));
        sentences.add(new Vector<>(Lists.newArrayList("the", "quick", "brown", "vox", "jumps", "over", "the", "lazy", "dog", "and")));
        sentences.add(new Vector<>(Lists.newArrayList("the", "quick", "brown", "vox", "jumps", "over", "the", "lazy", "dog", "and")));
        sentences.add(new Vector<>(Lists.newArrayList("the", "quick", "brown", "vox", "jumps", "over", "the", "lazy", "dog", "and")));
        sentences.add(new Vector<>(Lists.newArrayList("the", "quick", "brown", "vox", "jumps", "over", "the", "lazy", "dog", "and")));
        sentences.add(new Vector<>(Lists.newArrayList("the", "quick", "brown", "vox", "jumps", "over", "the", "lazy", "dog", "and")));
        sentences.add(new Vector<>(Lists.newArrayList("the", "quick", "brown", "vox", "jumps", "over", "the", "lazy", "dog", "and")));
        sentences.add(new Vector<>(Lists.newArrayList("the", "quick", "brown", "vox", "jumps", "over", "the", "lazy", "dog", "and")));
        ZealousWordEmbeddingTrainingStrategy trainingStrategy = new ZealousWordEmbeddingTrainingStrategy(ffn, bpTrain, embeddings, 1);
        trainingStrategy.train(sentences);
        return trainingStrategy.verifyGradient(DoubleMatrix.rand(501), DoubleMatrix.rand(1));
    }

}
