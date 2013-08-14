package com.github.alleveenstra.neuralwordembedding.tools.util;

import java.text.Normalizer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Sanitization {

    private static final Pattern NUMBERS = Pattern.compile("[0-9]");

    public static final Pattern DIACRITICS_AND_FRIENDS = Pattern.compile("[\\p{InCombiningDiacriticalMarks}\\p{IsLm}\\p{IsSk}]+");

    private static String stripDiacritics(String str) {
        str = Normalizer.normalize(str, Normalizer.Form.NFD);
        str = DIACRITICS_AND_FRIENDS.matcher(str).replaceAll("");
        return str;
    }

    public static String sanitizeWord(String input) {
        StringBuilder stringBuilder = new StringBuilder(input);
        final Matcher numbers = NUMBERS.matcher(input);
        while (numbers.find()) {
            stringBuilder.replace(numbers.start(), numbers.end(), "D");
        }
        final String result = stringBuilder.toString();
        if (result.equals(input)) {
            return input;
        }
        stringBuilder.insert(0, "*");
        stringBuilder.append("*");
        return stringBuilder.toString();
    }

    public static String sanitizeLine(String input) {
        input = input.replaceAll("\r", "");
        input = input.replaceAll("»", "");
        input = input.replaceAll("”", "\"");
        input = input.replaceAll("“", "\"");
        input = input.replaceAll("‘", "'");
        input = input.replaceAll("’", "'");
        input = input.replaceAll("²", "2");
        input = input.replaceAll("³", "3");
        input = input.replaceAll("…", "...");
        input = input.replaceAll("–", "-");
        input = input.replaceAll("—", "-");
        input = input.replaceAll("­", "-");
        input = input.replaceAll("×", "*");
        input = input.replaceAll("°", "DEG");
        input = input.replaceAll("˚", "DEG");
        input = input.replaceAll("º", "DEG");
        input = input.replaceAll("€", "EUR");
        input = input.replaceAll("£", "PND");
        input = input.replaceAll("«", "");
        input = stripDiacritics(input);
        input = input.replaceAll(" ", " "); // do not optimize
        input = input.replaceAll("'", ""); // remove 's
        input = input.replaceAll("\"", ""); // remove "s
        return input;
    }
}
