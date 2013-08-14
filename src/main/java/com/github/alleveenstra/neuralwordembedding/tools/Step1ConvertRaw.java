package com.github.alleveenstra.neuralwordembedding.tools;

import com.google.common.base.Joiner;
import com.github.alleveenstra.neuralwordembedding.tools.util.Parser;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.io.filefilter.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;

public class Step1ConvertRaw {
    private static final Logger log = LoggerFactory.getLogger(Step1ConvertRaw.class);

    private static final ForkJoinPool forkJoinPool = new ForkJoinPool(8);
    public static final String[] EMPTY_STRING_ARRAY = new String[0];

    public static void main(String[] args) {
        final String directory = "./data/raw";
        forkJoinPool.invoke(new FolderSearchTask(directory));
    }

    private static class ProcessFileTask extends RecursiveTask<String> {
        private File file;

        public ProcessFileTask(File file) {
            this.file = file;
        }

        @Override
        protected String compute() {
            long startTime = System.currentTimeMillis();
            System.out.println("Processing " + file.getAbsolutePath());
            final Parser parser = new Parser();
            FileWriter fileWriter = null;
            try {
                final String writeFilename = file.getAbsolutePath().replace(".txt", ".dataset").replace("/raw/", "/dataset/");
                final File writeFile = new File(writeFilename);
                if (!writeFile.exists()) {
                    writeFile.createNewFile();
                } else {
                    writeFile.delete();
                    writeFile.createNewFile();
                }
                fileWriter = new FileWriter(writeFile, true);

                final List<String> sentences = new ArrayList<>();
                int newlineCounter = 0;
                sentences.add("<s>");
                for (String line : loadFile(file)) {
                    if (line.isEmpty()) {
                        newlineCounter++;
                    } else {
                        if (newlineCounter > 2) {
                            sentences.add("</s>");
                            sentences.add("<s>");
                            sentences.addAll(parser.extractSentences(line));
                        } else {
                            sentences.addAll(parser.extractSentences(line));
                        }
                        newlineCounter = 0;
                    }
                }
                sentences.add("</s>");
                for (String sentence : sentences) {
                    final List<String> words = parser.tokenize(sentence);
                    final String newSentence = Joiner.on(" ").join(words.toArray(EMPTY_STRING_ARRAY));
                    fileWriter.write(newSentence + "\n");
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

        public FolderSearchTask(String directory) {
            this.directory = directory;
        }
        @Override
        protected String compute() {
            final List<RecursiveTask<String>> forks = new LinkedList<>();
            final Collection<File> files = (Collection<File>) FileUtils.listFiles(new File(directory), new SuffixFileFilter(".txt"), TrueFileFilter.TRUE);
            for (File file : files) {
                ProcessFileTask task = new ProcessFileTask(file);
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
}
