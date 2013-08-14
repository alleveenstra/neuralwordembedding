package com.github.alleveenstra.neuralwordembedding.neuralnetwork;

import com.google.common.collect.Lists;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ZealousWordEmbeddingTrainingStrategy {
    private final static Logger log = LoggerFactory.getLogger(ZealousWordEmbeddingTrainingStrategy.class);

    protected static final int EMBEDDING_SIZE = 50;

    protected double etaEmbedding = 0.000320;
    protected int windowSize = 10;
    protected int corruptElement = 9;

    protected ArrayList<String> words;

    protected final FeedForwardNetwork network;
    protected final BackPropagationTrainer trainer;
    protected final Map<String, DoubleMatrix> embeddings;

    private final Random random = new Random();

    private final int concurrency;
    private final ExecutorService executorService;
    private int next;
    private int[] counter;


    public ZealousWordEmbeddingTrainingStrategy(
            final FeedForwardNetwork network,
            final BackPropagationTrainer trainer,
            final Map<String, DoubleMatrix> embeddings,
            final int concurrency) {
        if (concurrency < 1) {
            throw new IllegalArgumentException("Concurrency needs to be at least 1");
        }
        this.network = network;
        this.trainer = trainer;
        this.embeddings = embeddings;
        this.words = new ArrayList<>(embeddings.keySet());
        this.concurrency = concurrency;
        this.executorService = Executors.newFixedThreadPool(this.concurrency);
        this.next = 0;
        this.counter = new int[this.concurrency];
    }

    public void setEta(double eta) {
        trainer.eta = eta;
    }

    public void setEtaEmbedding(double eta_embedding) {
        this.etaEmbedding = eta_embedding;
    }

    public double validate(List<Vector<String>> dataSet) {
        final Mean mean = new Mean();
        for (Vector<String> correctPoint : dataSet) {
            final Vector<String> corruptedPoint = (Vector<String>) correctPoint.clone();
            corruptedPoint.set(corruptElement, randomWord());
            final DoubleMatrix correctData = concatenateEmbeddings(correctPoint);
            final DoubleMatrix corruptedData = concatenateEmbeddings(corruptedPoint);
            final Score score = rankingCriterium(correctData, corruptedData);
            mean.increment(score.rank());
        }
        return mean.getResult();
    }

    // protected

    protected String randomWord() {
        return words.get(random.nextInt(words.size()));
    }

    protected DoubleMatrix concatenateEmbeddings(Vector<String> dataPoint) {
        DoubleMatrix result = null;
        for (int offset = 0; offset < windowSize; offset++) {
            final DoubleMatrix wordEmbedding = embeddings.get(dataPoint.get(offset));
            if (result == null) {
                result = wordEmbedding;
            } else {
                result = DoubleMatrix.concatHorizontally(result, wordEmbedding);
            }
        }
        return result;
    }

    protected Score rankingCriterium(DoubleMatrix correctData, DoubleMatrix corruptedData) {
        double correctScore = trainer.evaluate(correctData).scalar();
        double corruptedScore = trainer.evaluate(corruptedData).scalar();
        return new Score(correctScore, corruptedScore);
    }

    protected class Score {
        final double correct;
        final double corrupt;
        Score(double correct, double corrupt) {
            this.correct = correct;
            this.corrupt = corrupt;
        }
        double rank() {
            return Math.max(0.0, 1.0 - correct + corrupt);
        }
    }

    public void train(final List<Vector<String>> dataSet) {
        if (dataSet.size() == 0) {
            return; // easy training
        }
        final int subSetSize = (int) Math.ceil((double) dataSet.size() / (double) concurrency);
        final List<List<Vector<String>>> subSets = Lists.partition(dataSet, subSetSize);
        final CountDownLatch latch = new CountDownLatch(concurrency);
        for (int pid = 0; pid < this.concurrency; pid++) {
            executorService.execute(new MiniBatchTrainingTask(latch, pid, subSets.get(pid)));
        }
        try {
            latch.await();
        } catch (InterruptedException e) {
            log.error("interrupted!", e);
        }
    }

    public void shutdown() {
        executorService.shutdown();
    }

    private static class Context {
        private FeedForwardNetwork network;
        private BackPropagationTrainer trainer;
        public Context(final FeedForwardNetwork originNetwork, final BackPropagationTrainer originTrainer) {
            if (originNetwork.extraInput) {
                this.network = new FeedForwardNetwork(originNetwork.extraInput, originNetwork.inputSize - 1, originNetwork.hiddenShape);
            } else {
                this.network = new FeedForwardNetwork(originNetwork.extraInput, originNetwork.inputSize, originNetwork.hiddenShape);
            }
            this.trainer = new BackPropagationTrainer(network);

            // copy all parameters
            this.trainer.eta = originTrainer.eta;
            this.trainer.eta_bias = originTrainer.eta_bias;
            this.trainer.eta_L1 = originTrainer.eta_L1;
            this.trainer.eta_L2 = originTrainer.eta_L2;
            this.trainer.eta_momentum = originTrainer.eta_momentum;
            this.trainer.eta_decay = originTrainer.eta_decay;

            // inject all weights by reference, don't copy them
            for (int index = 1; index < this.network.weights.length; index++) {
                this.network.weights[index] = originNetwork.weights[index];
            }
        }
    }

    private class MiniBatchTrainingTask implements Runnable {
        private final List<Vector<String>> batch;
        private final Context context;
        private final int pid;

        private DoubleMatrix cumulativeWeightUpdate[] = null;
        private DoubleMatrix cumulativeBiasUpdate[] = null;
        private Map<String, DoubleMatrix> cumulativeEmbeddingUpdate;
        private CountDownLatch latch;

        public MiniBatchTrainingTask(CountDownLatch latch, int pid, List<Vector<String>> batch) {
            this.latch = latch;
            this.pid = pid;
            this.batch = batch;
            this.cumulativeEmbeddingUpdate = new HashMap<>();
            this.context = new Context(network, trainer);
        }

        @Override
        public void run() {
            try {
                for (final Vector<String> dataPoint : batch) {
                    final Vector<String> corruptedPoint = (Vector<String>) dataPoint.clone();
                    corruptedPoint.set(corruptElement, randomWord());
                    learn(dataPoint, corruptedPoint);
                    if (trylock(pid)) {
                        trainer.applyWeightUpdate(cumulativeBiasUpdate, cumulativeWeightUpdate);
                        cumulativeWeightUpdate = null;
                        cumulativeBiasUpdate = null;
                        for (Map.Entry<String, DoubleMatrix> entry : cumulativeEmbeddingUpdate.entrySet()) {
                            embeddings.get(entry.getKey()).addi(entry.getValue());
                        }
                        this.cumulativeEmbeddingUpdate.clear();
                        next(pid);
                    }
                }
            } finally {
                latch.countDown();
            }
        }

        private void learn(final Vector<String> correctPoint, final Vector<String> corruptedPoint) {
            final DoubleMatrix correctData = concatenateEmbeddings(correctPoint);
            final DoubleMatrix corruptedData = concatenateEmbeddings(corruptedPoint);
            double correctScore = this.context.trainer.evaluate(correctData).scalar();
            double corruptedScore = this.context.trainer.evaluate(corruptedData).scalar();
            double rankingCriterium = Math.max(0.0, 1.0 - correctScore + corruptedScore);
            if (rankingCriterium != 0.0) {
                double distance = (1 - (correctScore - corruptedScore)) / 2.0;
                double correctTarget = correctScore + distance;
                double corruptTarget = corruptedScore - distance;

                final DoubleMatrix correctUpdate = learnLocalSequential(correctData, DoubleMatrix.ones(1, 1).put(0, correctTarget));
                final DoubleMatrix corruptedUpdate = learnLocalSequential(corruptedData, DoubleMatrix.ones(1, 1).put(0, corruptTarget));

                updateLocalEmbedding(correctPoint, correctUpdate);
                updateLocalEmbedding(corruptedPoint, corruptedUpdate);
            }
        }

        private void updateLocalEmbedding(final Vector<String> point, final DoubleMatrix update) {
            final DoubleMatrix reshapedUpdate = update.reshape(EMBEDDING_SIZE, windowSize);
            for (int offset = 0; offset < windowSize; ++offset) {
                final String word = point.get(offset);
                final DoubleMatrix embeddingUpdate = reshapedUpdate.getColumn(offset);
                if (cumulativeEmbeddingUpdate.containsKey(word)) {
                    cumulativeEmbeddingUpdate.get(word).addi(embeddingUpdate);
                } else {
                    cumulativeEmbeddingUpdate.put(word, embeddingUpdate);
                }
            }
        }

        private DoubleMatrix learnLocalSequential(final DoubleMatrix inputs, final DoubleMatrix target) {
            if (target.length != this.context.network.outputSize) {
                throw new IllegalArgumentException("Target vector of incorrect size");
            }

            this.context.trainer.evaluate(inputs);
            final DoubleMatrix[] biasUpdate = this.context.trainer.calculateDeltas(target);
            final DoubleMatrix[] weightUpdate = this.context.trainer.calculateWeightUpdate();

            if (cumulativeWeightUpdate == null) {
                cumulativeWeightUpdate = weightUpdate;
                cumulativeBiasUpdate = biasUpdate;
            } else {
                for (int i = 1; i < weightUpdate.length; ++i) { // start at 1, not 0
                    cumulativeWeightUpdate[i].addi(weightUpdate[i]);
                    cumulativeBiasUpdate[i].addi(biasUpdate[i]);
                }
            }

            // specialz
            final DoubleMatrix error = this.context.network.deltas[1].mmul(this.context.network.weights[1].transpose());
            final DoubleMatrix deriv = this.context.network.activations[0].dup();
            this.context.trainer.derivation(deriv);
            this.context.network.deltas[0] = error.mul(deriv);
            this.context.network.deltas[0].muli(etaEmbedding);
            return this.context.network.deltas[0];
            // /specialz
        }
    }

    private boolean trylock(final int pid) {
        counter[pid] += 1;
        return next == pid;
    }

    private void next(final int pid) {
        counter[pid] = 0;
        next = argmax(counter);
    }

    private int argmax(final int[] counter) {
        int max = -1, maxIndex = -1;
        for (int i = 0; i < counter.length; ++i) {
            if (counter[i] > max) {
                max = counter[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public double verifyGradient(DoubleMatrix input, DoubleMatrix target) {
        double epsilon = 0.01;
        double total_difference = 0.0, n_differences = 0.0;
        for (int layer = 1; layer < network.nLayers; layer++) {
            for (int row = 0; row < network.weights[layer].getRows(); ++row) {
                for (int column = 0; column < network.weights[layer].getColumns(); ++column) {
                    final double weight = network.weights[layer].get(row, column);
                    network.weights[layer].put(row, column, weight + epsilon);
                    DoubleMatrix positiveOutput = trainer.evaluate(input);
                    double errorP1 = 0.5 * positiveOutput.squaredDistance(target);

                    network.weights[layer].put(row, column, weight - epsilon);
                    DoubleMatrix negativeOutput = trainer.evaluate(input);
                    double errorP2 = 0.5 * negativeOutput.squaredDistance(target);

                    network.weights[layer].put(row, column, weight);

                    trainer.evaluate(input);

                    double approx = (errorP1 - errorP2) / (epsilon * 2.0);

                    trainer.calculateDeltas(target); // FAK!

                    double gradient = -(network.activations[layer - 1].get(0, row) * network.deltas[layer].get(0, column));

                    total_difference += Math.abs(gradient - approx);
                    n_differences++;
                }
            }
        }
        return total_difference / n_differences;
    }
}
