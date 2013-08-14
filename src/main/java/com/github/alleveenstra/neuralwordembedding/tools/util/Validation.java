package com.github.alleveenstra.neuralwordembedding.tools.util;

import java.text.Normalizer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Validation {
    private static final Pattern ALLOWED_PATTERN = Pattern.compile("[A-Za-z0-9_=!@#%&,/:;\"'`<> \\|\\}\\{\\?\\.\\*\\(\\)\\^\\$\\-\\+\\]\\[\\\\]+");
    private static final Pattern DIACRITIC_MARKS = Pattern.compile("\\p{InCombiningDiacriticalMarks}+");

    public static boolean isValidSentence(final String sentence) {
        final String normalized = Normalizer.normalize(sentence, Normalizer.Form.NFD);
        final String cleaned = DIACRITIC_MARKS.matcher(normalized).replaceAll("");
        final Matcher matcher = ALLOWED_PATTERN.matcher(cleaned);
        return matcher.matches();
    }
}
