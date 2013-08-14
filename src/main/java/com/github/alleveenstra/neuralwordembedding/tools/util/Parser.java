package com.github.alleveenstra.neuralwordembedding.tools.util;

import com.google.common.io.Resources;
import opennlp.tools.sentdetect.SentenceDetector;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import org.apache.commons.io.IOUtils;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class Parser {
    private static final String SENTENCE_MODEL = "nl-sent.bin";
    private static final String TOKENIZER_MODEL = "nl-token.bin";

    private SentenceDetector sentenceDetector;
    private Tokenizer tokenizer;

    public Parser() {
        InputStream sentenceModelStream = null;
        InputStream modelToken = null;
        try {
            sentenceModelStream = Resources.getResource(SENTENCE_MODEL).openStream();
            final SentenceModel sentenceModel = new SentenceModel(sentenceModelStream);
            sentenceDetector = new SentenceDetectorME(sentenceModel);
            modelToken =  Resources.getResource(TOKENIZER_MODEL).openStream();
            TokenizerModel tokenizerModel = new TokenizerModel(modelToken);
            tokenizer = new TokenizerME(tokenizerModel);
        } catch (IOException exception) {
            throw new IllegalStateException(exception);
        } finally {
            IOUtils.closeQuietly(sentenceModelStream);
            IOUtils.closeQuietly(modelToken);
        }
    }

    public List<String> extractSentences(String input) {
        final List<String> processedSentences = new ArrayList<>();
        final String sentences[] = sentenceDetector.sentDetect(input);
        for (int i = 0; i < sentences.length; i++) {
            final String sanitizedLine = Sanitization.sanitizeLine(sentences[i]);
            if (Validation.isValidSentence(sanitizedLine)) {
                processedSentences.add(sanitizedLine);
            }
        }
        return processedSentences;
    }

    public List<String> tokenize(String sentence) {
        final String[] words = tokenizer.tokenize(sentence);
        final List<String> tokens = new ArrayList<>();
        for (int i = 0; i < words.length; i++) {
            final String word = words[i].trim();
            if (!word.isEmpty()) {
                final String lowerCase = word.toLowerCase();
                final String sanitizedWord = Sanitization.sanitizeWord(lowerCase);
                if (sanitizedWord.length() > 0) {
                    tokens.add(sanitizedWord);
                }
            }
        }
        return tokens;
    }
}
