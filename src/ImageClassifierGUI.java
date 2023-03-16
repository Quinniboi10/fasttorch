import java.awt.BorderLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.BufferedReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import javax.imageio.ImageIO;
import javax.swing.*;

public class ImageClassifierGUI {

    private static JFileChooser fileChooser = new JFileChooser();

    private static JTextArea outputTextArea = new JTextArea(1, 20);

    public static void main(String[] args) {
        // Create the main frame
        JFrame frame = new JFrame("Image Classifier");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(400, 250);

        // Create the image file upload button and preview panel
        JButton imageUploadButton = new JButton("Upload Image");
        JPanel imagePreviewPanel = new JPanel();
        imagePreviewPanel.setLayout(new BorderLayout());
        JLabel imagePreviewLabel = new JLabel();
        imagePreviewPanel.add(imagePreviewLabel, BorderLayout.CENTER);

        // Set action listener for the image upload button
        imageUploadButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                // Open file chooser dialog
                int result = fileChooser.showOpenDialog(frame);
                if (result == JFileChooser.APPROVE_OPTION) {
                    // Get the selected image file
                    File selectedFile = fileChooser.getSelectedFile();
                    // Display the selected image file as a preview
                    try {
                        ImageIcon imageIcon = new ImageIcon(ImageIO.read(selectedFile));
                        imagePreviewLabel.setIcon(imageIcon);
                    } catch (IOException ex) {
                        ex.printStackTrace();
                    }
                }
            }
        });

        // Create the classify button
        JButton classifyButton = new JButton("Classify");
        classifyButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                changeText("Thinking...");
                // Get the selected image file path
                String imagePath = fileChooser.getSelectedFile().getAbsolutePath();

                // Run the classifier code
                System.out.println("Thinking...");
                String currentDirectory = System.getProperty("user.dir");
                System.out.println("Current directory: " + currentDirectory);
                String[] command = {"C:\\Users\\qitag\\AppData\\Local\\Programs\\Python\\Python37\\python.exe \"C:\\Users\\qitag\\IdeaProjects\\ScienceGUI\\src\\fastai_classifier.py\" --model_path \"C:\\Users\\qitag\\IdeaProjects\\ScienceGUI\\src\\export bestmodel on resnet18 - plants.pkl\" --image_path " + imagePath};
                System.out.println(command[0]);
                try {
                    Process process = Runtime.getRuntime().exec(command[0]);
                    process.waitFor();
                    BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
                    String line;
                    while ((line = reader.readLine()) != null) {
                        System.out.println(line);
                    }
                    Path outputFilePath = Paths.get("output.txt");
                    String classification = Files.readString(outputFilePath).trim();
                    changeText(classification);
                } catch (IOException | InterruptedException ex) {
                    ex.printStackTrace();
                }
            }
        });

        // Create the output text area and classify button panel
        JPanel outputPanel = new JPanel(new BorderLayout());
        outputPanel.add(outputTextArea, BorderLayout.CENTER);
        outputPanel.add(classifyButton, BorderLayout.EAST);

        // Create the main panel
        JPanel mainPanel = new JPanel(new BorderLayout());
        mainPanel.add(imageUploadButton, BorderLayout.NORTH);
        mainPanel.add(imagePreviewPanel, BorderLayout.CENTER);
        mainPanel.add(outputPanel, BorderLayout.SOUTH);

        // Add the main panel to the frame
        frame.add(mainPanel);

        // Show the frame
        frame.setVisible(true);
    }

    public static void changeText(String text) {
        outputTextArea.setText(text);
    }
}