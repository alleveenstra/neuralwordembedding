package com.github.alleveenstra.neuralwordembedding.tools;

import com.google.common.base.Splitter;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.io.filefilter.SuffixFileFilter;
import org.apache.commons.io.filefilter.TrueFileFilter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;

public class Step3Enumerate {

    private static final Logger log = LoggerFactory.getLogger(Step3Enumerate.class);

    private static final ForkJoinPool forkJoinPool = new ForkJoinPool();

    public static void main(String[] args) throws IOException {
        Map<String, VocabularyItem> vocabulary = loadVocabulary("./data/vocab.txt");
        final String directory = "./data/dataset";
        forkJoinPool.invoke(new FolderSearchTask(directory, vocabulary));
    }

    private static Map<String, VocabularyItem> loadVocabulary(String filename) throws IOException {
        Map<String, VocabularyItem> vocabulary = new HashMap<>();
        File file = new File(filename);
        FileInputStream inputStream = new FileInputStream(file);
        InputStreamReader inputStreamReader = new InputStreamReader(inputStream);
        final BufferedReader bufferedReader = new BufferedReader(inputStreamReader);
        String line = bufferedReader.readLine();
        while (line != null) {
            final String[] split = line.split("[\t ]+");
            if (split.length >= 3) {
                vocabulary.put(split[0], new VocabularyItem(Long.parseLong(split[1]), Long.parseLong(split[2])));
            }
            line = bufferedReader.readLine();
        }
        IOUtils.closeQuietly(inputStream);
        return vocabulary;
    }

    private static class ProcessFileTask extends RecursiveTask<String> {
        private File file;
        private Map<String, VocabularyItem> vocabulary;

        public ProcessFileTask(File file, Map<String, VocabularyItem> vocabulary) {
            this.file = file;
            this.vocabulary = vocabulary;
        }

        @Override
        protected String compute() {
            long startTime = System.currentTimeMillis();
            System.out.println("Processing " + file.getAbsolutePath());
            FileWriter fileWriter = null;
            try {
                final String writeFilename = file.getAbsolutePath().replace(".dataset", ".numbered").replace("/dataset/", "/numbered/");
                final File writeFile = new File(writeFilename);
                if (!writeFile.exists()) {
                    writeFile.createNewFile();
                } else {
                    writeFile.delete();
                    writeFile.createNewFile();
                }
                fileWriter = new FileWriter(writeFile, true);
                for (String line : loadFile(file)) {
                    if (line.startsWith("<s>")) {
                        line = "<s>";
                    }
                    for (String word : Splitter.on(" ").split(line)) {
                        if (vocabulary.containsKey(word)) {
                            fileWriter.write(vocabulary.get(word).index + "\n");
                        } else {
                            fileWriter.write("1\n"); // unknown
                        }
                        if ("</s>".equals(word)) {
                            fileWriter.write("eeeoddd\n");
                        }
                    }
                }
            } catch(Exception e) {
                log.error("Unable to create new file", e);
            } finally {
                IOUtils.closeQuietly(fileWriter);
            }
            long endTime = System.currentTimeMillis();
            long totalTime = endTime - startTime;
            log.info("Ran task for {}", totalTime);
            return "ready";
        }
    }

    private static class FolderSearchTask extends RecursiveTask<String> {
        private String directory;
        private Map<String, VocabularyItem> vocabulary;

        public FolderSearchTask(String directory, Map<String, VocabularyItem> vocabulary) {
            this.directory = directory;
            this.vocabulary = vocabulary;
        }
        @Override
        protected String compute() {
            final List<RecursiveTask<String>> forks = new LinkedList<>();
            final Collection<File> files = (Collection<File>) FileUtils.listFiles(new File(directory), new SuffixFileFilter(".dataset"), TrueFileFilter.TRUE);
            for (File file : files) {
                ProcessFileTask task = new ProcessFileTask(file, vocabulary);
                forks.add(task);
                task.fork();
            }
            for (RecursiveTask<String> task : forks) {
                task.join();
            }
            return "ready";
        }
    }

    private static List<String> loadFile(File file) {
        FileInputStream fileInputStream = null;
        try {
            fileInputStream = new FileInputStream(file);
            return IOUtils.readLines(fileInputStream);
        } catch (FileNotFoundException e) {
            log.error("File not found", e);
        } catch (IOException e) {
            log.error("Can not read file", e);
        } finally {
            IOUtils.closeQuietly(fileInputStream);
        }
        return null;
    }

    static class VocabularyItem {
        long index;
        long occurrences;
        VocabularyItem(long index, long occurrences) {
            this.index = index;
            this.occurrences = occurrences;
        }
    }

}
