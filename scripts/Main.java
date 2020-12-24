import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;

public class Main {
    public static void main(String[] args) throws IOException {
        File f = new File("image_labels.csv");
        BufferedReader br = new BufferedReader(new FileReader(f));
        String line = br.readLine();
        HashMap<String, ArrayList<String>> map = new HashMap<>();
        while (!"".equals(line) && line != null) {
            String[] splits = line.split(",");
            String imageLabel = splits[0].trim();
            String[] diagnosisList = splits[1].trim().replace(" ", "_").split("\\|");
            for (String diagnosis : diagnosisList) {
                if (map.containsKey(diagnosis)) {
                    ArrayList<String> list = map.get(diagnosis);
                    list.add(imageLabel);
                } else {
                    ArrayList<String> list = new ArrayList<>();
                    list.add(imageLabel);
                    map.put(diagnosis, list);
                }
            }

            line = br.readLine();
        }

        // create script to organize files
        StringBuilder mkdirScript = new StringBuilder();
        StringBuilder cpScript = new StringBuilder();
        for (String s : map.keySet()) {
            mkdirScript.append("mkdir ").append(s).append(";\n");
            ArrayList<String> images = map.get(s);
            for (String image : images) {
                cpScript.append("cp all_images/").append(image).append(" ").append(s).append(";\n");
            }
        }

        BufferedWriter bw = new BufferedWriter(new FileWriter(new File("cpScript.sh")));
        bw.write(String.valueOf(cpScript));
        bw.flush();
        bw.close();
    }
}