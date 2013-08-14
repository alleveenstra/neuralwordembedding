package com.github.alleveenstra.neuralwordembedding.tools.training;

import com.github.alleveenstra.neuralwordembedding.neuralnetwork.FeedForwardNetwork;
import org.apache.commons.io.IOUtils;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;

public class Model implements Serializable {
    private static final long serialVersionUID = 2239967989698997387L;

    private static final Logger log = LoggerFactory.getLogger(Training.class);

    protected final Map<String, DoubleMatrix> embeddings;
    protected final FeedForwardNetwork network;

    protected Model(Map<String, DoubleMatrix> embeddings, FeedForwardNetwork network) {
        this.embeddings = embeddings;
        this.network = network;
    }

    public static Model load(File file) {
        FileInputStream fileInputStream = null;
        ObjectInputStream objectInputStream = null;
        try {
            fileInputStream = new FileInputStream(file);
            objectInputStream = new ObjectInputStream(fileInputStream);
            final Object readObject = objectInputStream.readObject();
            if (readObject instanceof Model) {
                return (Model)readObject;
            }
        } catch (ClassNotFoundException | IOException e) {
            log.error("Unable to read object", e);
        } finally {
            IOUtils.closeQuietly(fileInputStream);
            IOUtils.closeQuietly(objectInputStream);
        }
        return null;
    }

    public void save(String fileName) {
        log.info("Model.save({})", fileName);
        final File embeddingsFile = new File(fileName);
        FileOutputStream fileOutputStream = null;
        ObjectOutputStream objectOutputStream = null;
        try {
            if (!embeddingsFile.exists()) {
                log.info("Create new file...");
                embeddingsFile.createNewFile();
            }
            fileOutputStream = new FileOutputStream(embeddingsFile);
            objectOutputStream = new ObjectOutputStream(fileOutputStream);
            log.info("Writing object...");
            objectOutputStream.writeObject(this);
            log.info("Object written...");
        } catch (FileNotFoundException e) {
            log.error("No such file", e);
        } catch (IOException e) {
            log.error("Unable to save file", e);
        } finally {
            IOUtils.closeQuietly(fileOutputStream);
            IOUtils.closeQuietly(objectOutputStream);
        }
    }

    private class WordDistance implements Comparable<WordDistance> {
        String word;
        double distance;

        public WordDistance(String word, double distance) {
            this.word = word;
            this.distance = distance;
        }

        @Override
        public int compareTo(WordDistance o) {
            if (this.distance == o.distance) {
                return 0;
            }
            return this.distance < o.distance ? -1 : 1;
        }
    }

    public List<String> findCloseWords(String query, int count) {
        if (!embeddings.containsKey(query)) {
            return Collections.emptyList();
        }
        List<WordDistance> distances = new ArrayList<>();
        final DoubleMatrix queryVector = embeddings.get(query);
        for (Map.Entry<String, DoubleMatrix> entry : embeddings.entrySet()) {
            final DoubleMatrix row = entry.getValue();
            final double dist = row.squaredDistance(queryVector);
            distances.add(new WordDistance(entry.getKey(), dist));
        }
        Collections.sort(distances);
        final List<WordDistance> list = distances.subList(0, count);
        List<String> closeWords = new ArrayList<String>();
        for (WordDistance wordDistance : list) {
            closeWords.add(wordDistance.word);
        }
        return closeWords;
    }
}
