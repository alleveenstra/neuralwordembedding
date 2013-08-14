package com.github.alleveenstra.neuralwordembedding.tools.training;

import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.github.alleveenstra.neuralwordembedding.neuralnetwork.BackPropagationTrainer;
import com.github.alleveenstra.neuralwordembedding.neuralnetwork.FeedForwardNetwork;
import com.github.alleveenstra.neuralwordembedding.neuralnetwork.ZealousWordEmbeddingTrainingStrategy;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.io.filefilter.SuffixFileFilter;
import org.apache.commons.io.filefilter.TrueFileFilter;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;

public class Training {
    private static final Logger log = LoggerFactory.getLogger(Training.class);

    private String dataSetDirectory;
    private static final String RARE_WORD = "*RARE*";
    private Map<String, Integer> vocabulary;

    private List<String> stopWords = new ImmutableList.Builder<String>()
            .add("<s>", "</s>", "\"", "'", "(", ")", ",", "-", ".", ":", ";", "?", "!", "[", "]", "{", "}", "UUUNKKK", "aan", "achter", "alle", "alleen", "als", "andere", "anders", "ben", "bij", "bijna", "binnen", "d'r", "daar", "dan", "dat", "de", "den", "der", "des", "deed", "deze", "die", "dit", "doe", "doen", "dl", "echter", "een", "eerder", "elke", "en", "enige", "enkele", "enz", "er", "ervan", "etc", "evenmin", "haar", "hare", "hen", "het", "hierin", "hij", "hoe", "hun", "hunne", "iedere", "ik", "in", "inzake", "is", "ja", "je", "jouw", "jouwe", "juist", "jullie", "kan", "kun", "kon", "laat", "maar", "me", "meest", "met", "mijn", "mijne", "minst", "moet", "na", "nabij", "nee", "niet", "noch", "nog", "of", "om", "omdat", "onder", "ons", "onze", "ooit", "ook", "op", "over", "overheen", "sinds", "sommige", "te", "tegen", "ten", "ter", "tijdens", "tot", "uit", "uw", "uwe", "vaak", "van", "voor", "waar", "waarom", "wanneer", "waren", "was", "wat", "welke", "wie", "wij", "wilden", "willen", "z'n", "ze", "zij", "zijn", "zijne", "zo", "zou")
            .build();

    private final int EMBEDDING_SIZE   = 50;
    private final int WINDOW_SIZE      = 10;
    private final int HIDDEN_SIZE      = 100;

    private Model model;

    private BackPropagationTrainer trainer;
    private ZealousWordEmbeddingTrainingStrategy strategy;

    public Training(String vocabularyFile, String dataSetDirectory, String modelPath, int concurrency) {
        final File modelFile = new File(modelPath);
        try {
            this.dataSetDirectory = dataSetDirectory;
            this.vocabulary = Vocabulary.load(vocabularyFile);
            if (!modelFile.exists()) {
                final Map<String, DoubleMatrix> embeddings = initializeEmbeddings(vocabulary);
                final FeedForwardNetwork network = new FeedForwardNetwork(EMBEDDING_SIZE * WINDOW_SIZE, HIDDEN_SIZE, 1);
                this.model = new Model(embeddings, network);
            } else {
                final Model loadedModel = Model.load(modelFile);
                if (loadedModel == null) {
                    throw new IllegalStateException("Failed to load specified model.");
                }
                this.model = loadedModel;
            }
            trainer = new BackPropagationTrainer(this.model.network);
            strategy = new ZealousWordEmbeddingTrainingStrategy(this.model.network, trainer, this.model.embeddings, concurrency);
        } catch (IOException e) {
            throw new IllegalStateException("Unable to load vocabulary", e);
        }
    }
    
    public void trainOneEpoch(double eta, double etaEmbedding) {
        final Collection<File> files = (Collection<File>) FileUtils.listFiles(new File(dataSetDirectory), new SuffixFileFilter(".dataset"), TrueFileFilter.TRUE);
        for (File file : files) {
            strategy.setEta(eta);
            strategy.setEtaEmbedding(etaEmbedding);
            log.info("processing file {}", file.getAbsolutePath());
            final List<Vector<String>> dataSet = extractDataSet(file);
            strategy.train(dataSet);
        }
    }

    private Map<String, DoubleMatrix> initializeEmbeddings(Map<String, Integer> vocabulary) {
        Map<String, DoubleMatrix> embeddings = new HashMap<>();
        for (Map.Entry<String, Integer> entry : vocabulary.entrySet()) {
            embeddings.put(entry.getKey(), DoubleMatrix.rand(1, EMBEDDING_SIZE).min(0.5));
        }
        return embeddings;
    }

    private List<Vector<String>> extractDataSet(File file) {
        final List<Vector<String>> dataSet = new ArrayList<>();
        FileInputStream fileInputStream = null;
        try {
            fileInputStream = new FileInputStream(file);
            final List<String> sentences = IOUtils.readLines(fileInputStream);
            final List<String> words = new ArrayList<>();
            for (String sentence : sentences) {
                if ("<s>".equals(sentence)) {
                    appendWords(words, dataSet);
                    words.clear();
                } else {
                    for (String word : Splitter.on(" ").omitEmptyStrings().split(sentence)) {
                        if (vocabulary.containsKey(word) && !stopWords.contains(word)) {
                            words.add(word);
                        }
                    }
                }
            }
            appendWords(words, dataSet);
        } catch (IOException e) {
            log.error("Unable to read file", e);
        } finally {
            IOUtils.closeQuietly(fileInputStream);
        }
        return dataSet;
    }

    private void appendWords(List<String> words, List<Vector<String>> dataSet) {
        if (words.size() >= WINDOW_SIZE) {
            for (int start = 0; start < words.size() - WINDOW_SIZE; start++) {
                final Vector<String> dataPoint = new Vector<>();
                for (int offset = 0; offset < WINDOW_SIZE; ++offset) {
                    dataPoint.add(mapWord(words.get(start + offset)));
                }
                dataSet.add(dataPoint);
            }
        }
    }

    private String mapWord(String data) {
        if (this.model.embeddings.containsKey(data)) {
            return data;
        }
        return RARE_WORD;
    }

    public void saveModel(String modelFile) {
        this.model.save(modelFile);
    }

    public void shutdown() {
        strategy.shutdown();
    }

    public double validate() {
        return validate(dataSetDirectory);
    }

    public double validate(String directory) {
        Mean mean = new Mean();
        final Collection<File> files = (Collection<File>) FileUtils.listFiles(new File(directory), new SuffixFileFilter(".dataset"), TrueFileFilter.TRUE);
        for (File file : files) {
            final List<Vector<String>> dataSet = extractDataSet(file);
            mean.increment(strategy.validate(dataSet));
        }
        return mean.getResult();
    }
}
