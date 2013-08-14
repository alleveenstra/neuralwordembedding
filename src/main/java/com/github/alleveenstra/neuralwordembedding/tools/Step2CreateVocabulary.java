package com.github.alleveenstra.neuralwordembedding.tools;

import com.github.alleveenstra.neuralwordembedding.tools.util.Parser;
import com.github.alleveenstra.neuralwordembedding.tools.util.Sanitization;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.io.filefilter.TrueFileFilter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;
import java.util.concurrent.atomic.AtomicInteger;

public class Step2CreateVocabulary {
    private static final Logger log = LoggerFactory.getLogger(Step2CreateVocabulary.class);

    private static final ForkJoinPool forkJoinPool = new ForkJoinPool();

    public static void main(String[] args) throws FileNotFoundException, UnsupportedEncodingException {
        final Map<String, AtomicInteger> vocabulary = forkJoinPool.invoke(new FolderSearchTask("./data/dataset"));
        writeToVocab(vocabulary, "./data/vocab.txt");
    }

    private static void writeToVocab(Map<String, AtomicInteger> vocabulary, String filename) throws FileNotFoundException, UnsupportedEncodingException {
        PrintWriter writer = new PrintWriter(filename, "UTF-8");
        writer.printf("UUUNKKK\t1\t1\n");
        int counter = 2;
        for (Map.Entry<String, AtomicInteger> entry : vocabulary.entrySet()) {
            writer.printf("%s\t%d\t%d\n", entry.getKey(), counter, entry.getValue().get());
            counter++;
        }
    }

    private static class ProcessFileTask extends RecursiveTask<Map<String, AtomicInteger>> {
        private File file;

        public ProcessFileTask(File file) {
            this.file = file;
        }
        @Override
        protected Map<String, AtomicInteger> compute() {
            long startTime = System.currentTimeMillis();
            System.out.println("Processing " + file.getAbsolutePath());
            final Map<String, AtomicInteger> vocabulary = new HashMap<>();
            try {
                final Parser parser = new Parser();
                appendWordsFromFile(parser, file, vocabulary);
            } catch (Exception e) {
                log.error("Pool fault", e);
            }
            long endTime = System.currentTimeMillis();
            long totalTime = endTime - startTime;
            System.out.println("Ran task for " + totalTime);
            return vocabulary;
        }
    }

    private static class FolderSearchTask extends RecursiveTask<Map<String, AtomicInteger>> {
        private String directory;

        public FolderSearchTask(String directory) {
            this.directory = directory;
        }
        @Override
        protected Map<String, AtomicInteger> compute() {
            final Map<String, AtomicInteger> vocabulary = new HashMap<>();
            final List<RecursiveTask<Map<String, AtomicInteger>>> forks = new LinkedList<>();
            final Collection<File> files = (Collection<File>) FileUtils.listFiles(new File(directory), TrueFileFilter.TRUE, TrueFileFilter.TRUE);
            for (File file : files) {
                ProcessFileTask task = new ProcessFileTask(file);
                forks.add(task);
                task.fork();
            }
            for (RecursiveTask<Map<String, AtomicInteger>> task : forks) {
                final Map<String, AtomicInteger> subResult = task.join();
                for (Map.Entry<String, AtomicInteger> entry : subResult.entrySet()) {
                    if (vocabulary.containsKey(entry.getKey())) {
                        vocabulary.get(entry.getKey()).addAndGet(entry.getValue().intValue());
                    } else {
                        vocabulary.put(entry.getKey(), entry.getValue());
                    }
                }
            }
            return vocabulary;
        }
    }

    private static void appendWordsFromFile(Parser parser, File file, Map<String, AtomicInteger> vocabulary) {
        FileInputStream fileInputStream = null;
        InputStreamReader inputStreamReader = null;
        BufferedReader bufferedReader = null;
        try {
            fileInputStream = new FileInputStream(file);
            inputStreamReader = new InputStreamReader(fileInputStream);
            bufferedReader = new BufferedReader(inputStreamReader);
            String line = bufferedReader.readLine();
            while (line != null) {
                line = Sanitization.sanitizeLine(line);
                final List<String> sentences = parser.extractSentences(line);
                for (String sentence : sentences) {
                    appendWordsFromString(parser, sentence, vocabulary);
                }
                line = bufferedReader.readLine();
            }
        } catch (FileNotFoundException e) {
            log.error("File not found", e);
        } catch (IOException e) {
            log.error("Can not read file", e);
        } finally {
            IOUtils.closeQuietly(bufferedReader);
            IOUtils.closeQuietly(inputStreamReader);
            IOUtils.closeQuietly(fileInputStream);
        }
    }

    private static void appendWordsFromString(Parser parser, String input, Map<String, AtomicInteger> vocabulary) {
        final List<String> sentences = parser.extractSentences(input);
        for (String sentence : sentences) {
            final List<String> tokens = parser.tokenize(sentence);
            for (String word : tokens) {
                if (vocabulary.containsKey(word)) {
                    vocabulary.get(word).addAndGet(1);
                } else {
                    vocabulary.put(word, new AtomicInteger(1));
                }
            }
        }
    }

}
