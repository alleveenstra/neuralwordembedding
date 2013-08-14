package com.github.alleveenstra.neuralwordembedding.tools;

import com.github.alleveenstra.neuralwordembedding.tools.training.Model;
import com.github.alleveenstra.neuralwordembedding.tools.training.Training;
import org.apache.commons.cli.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.List;

public class Step4LearnWordEmbeddings {
    private static final Logger log = LoggerFactory.getLogger(Step4LearnWordEmbeddings.class);

    public static void main(String[] args) {
        final Options options = new Options();
        options.addOption("d", "dataset", true, "The dataset directory containing '.dataset' files (required)");
        options.addOption("v", "vocabulary", true, "The vocabulary file (required)");
        options.addOption("c", "concurrency", true, "The number of processes");
        options.addOption("read", true, "The model file to read");
        options.addOption("save", true, "The model file to save");
        options.addOption("validate", true, "Validate on the .dataset files in this directory");
        options.addOption("epochs", true, "Number of epochs to run for, default: 1");
        options.addOption("s", "start", true, "Start epoch, default: 0");
        options.addOption("l", "learning-rate", true, "Learning rate for the neural network, default: 0.000001");
        options.addOption("m", "embedding-rate", true, "Learning rate for the word embeddings, default: 0.000320");
        options.addOption("r", "decrease-rate", true, "Rate to decrease learning rates, expression: E(t) = E(0) / (1 + r * t), default: 0");
        options.addOption("help", false, "Shows this help");
        final CommandLineParser parser = new GnuParser();
        try {
            final CommandLine cmd = parser.parse(options, args);
            if (cmd.hasOption("help") || cmd.getArgs().length == 0) {
                help(options);
                return;
            }
            int epochs = 1, startEpoch = 0;
            int concurrency = defaultConcurrency();
            String readFileName = "no-such-file", saveFileName = null, dataSet = null, vocabulary = null, validate = null;
            double eta = 0.000001, etaEmbedding = 0.000320, decreaseRate = 0.0;
            if (cmd.hasOption("epochs")) {
                epochs = Integer.parseInt(cmd.getOptionValue("epochs"));
            }
            if (cmd.hasOption("concurrency")) {
                concurrency = Integer.parseInt(cmd.getOptionValue("concurrency"));
            }
            if (cmd.hasOption("start")) {
                startEpoch = Integer.parseInt(cmd.getOptionValue("start"));
            }
            if (cmd.hasOption("read")) {
                readFileName = cmd.getOptionValue("read");
            }
            if (cmd.hasOption("save")) {
                saveFileName = cmd.getOptionValue("save");
            }
            if (cmd.hasOption("validate")) {
                validate = cmd.getOptionValue("validate");
            }
            if (cmd.hasOption("dataset")) {
                dataSet = cmd.getOptionValue("dataset");
            }
            if (cmd.hasOption("vocabulary")) {
                vocabulary = cmd.getOptionValue("vocabulary");
            }
            if (cmd.hasOption("learning-rate")) {
                eta = Double.parseDouble(cmd.getOptionValue("learning-rate"));
            }
            if (cmd.hasOption("embedding-rate")) {
                etaEmbedding = Double.parseDouble(cmd.getOptionValue("embedding-rate"));
            }
            if (cmd.hasOption("decrease-rate")) {
                decreaseRate = Double.parseDouble(cmd.getOptionValue("decrease-rate"));
            }
            final String action = cmd.getArgs()[0];
            switch (action) {
                case "learn":
                    if (dataSet == null || vocabulary == null || saveFileName == null) {
                        log.error("The dataset, vocabulary and save parameters are mandatory for learning.");
                        return;
                    }
                    learn(epochs, startEpoch, readFileName, saveFileName, validate, dataSet, vocabulary, eta, etaEmbedding, decreaseRate, concurrency);
                    log.info("Finished learning...");
                    break;
                case "search":
                    if (cmd.getArgs().length == 1) {
                        log.error("Please specify a search term");
                        return;
                    }
                    search(cmd.getArgs()[1], readFileName);
                    break;
                case "validate":
                    if (dataSet == null || readFileName == null || vocabulary == null) {
                        log.error("The dataset, read and vocabulary parameters is required for validation.");
                        return;
                    }
                    validate(dataSet, readFileName, vocabulary);
                    break;
                default:
                    log.error("Unknown action \"{}\"", action);
                    help(options);
                    return;
            }
        } catch (ParseException e) {
            log.error("Unable to parse command line arguments", e);
            return;
        }
    }

    private static int defaultConcurrency() {
        final int nProcs = Runtime.getRuntime().availableProcessors();
        return nProcs == 1 ? 1 : nProcs - 1;
    }

    private static void search(String term, String readFileName) {
        Model model = Model.load(new File(readFileName));
        model.findCloseWords(term, 10);
        final List<String> closeWords = model.findCloseWords(term, 10);
        for (String closeWord : closeWords) {
            System.out.println(closeWord);
        }
    }

    private static void learn(int epochs, int startEpoch, String readFileName, String saveFileName, String validate, String dataset, String vocabulary, double eta0, double etaEmbedding0, double decreaseRate, int concurrency) {
        long start, stop;
        final Training training = new Training(vocabulary, dataset, readFileName, concurrency);
        start = System.currentTimeMillis();
        double eta, etaEmbedding;
        for (int i = startEpoch; i < startEpoch + epochs; i++) {
            eta = eta0 / (1.0 + decreaseRate * i);
            etaEmbedding = etaEmbedding0 / (1.0 + decreaseRate * i);
            log.info(String.format("*** epoch %d eta %.8f eta embedding %.8f concurrency %d", i, eta, etaEmbedding, concurrency));
            training.trainOneEpoch(eta, etaEmbedding);
            if (validate != null) {
                log.info(String.format("Validation %.08f", training.validate(validate)));
            }
        }
        training.shutdown();
        stop = System.currentTimeMillis();
        System.out.println(String.format("Training took %.2f sec.", (stop - start) / 1000.0));
        if (saveFileName != null) {
            log.info("Saving to {}...", saveFileName);
            training.saveModel(saveFileName);
        }
    }

    private static void validate(String dataset, String readFileName, String vocabulary) {
        final Training training = new Training(vocabulary, dataset, readFileName, 1);
        double meanError = training.validate();
        training.shutdown();
        System.out.println(String.format("The mean error is %.2f", meanError));
    }

    private static void help(Options options) {
        HelpFormatter formatter = new HelpFormatter();
        formatter.printHelp("Step4LearnWordEmbeddings [learn | validate | search *term*]", options);
    }
}
