package com.github.alleveenstra.neuralwordembedding.tools.training;

import org.apache.commons.io.IOUtils;

import java.io.*;
import java.util.*;

public class Vocabulary {
    public static Map<String, Integer> load(String vocabularyFile) throws IOException {
        final Map<String, Integer> vocabulary = new HashMap<>();
        File file = new File(vocabularyFile);
        FileInputStream inputStream = new FileInputStream(file);
        InputStreamReader inputStreamReader = new InputStreamReader(inputStream);
        final BufferedReader bufferedReader = new BufferedReader(inputStreamReader);
        String line = bufferedReader.readLine();
        while (line != null) {
            final String[] split = line.trim().split("[\t ]+");
            if (split.length >= 2) {
                final int index = Integer.parseInt(split[1]);
                vocabulary.put(split[0], index);
            }
            line = bufferedReader.readLine();
        }
        IOUtils.closeQuietly(inputStream);
        return vocabulary;
    }
}
