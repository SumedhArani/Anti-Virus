import java.net.*;
import java.io.*;
import java.util.*;


public class BinToHex
{
	static LinkedHashMap<String,Integer> dictionary = new LinkedHashMap<String,Integer>();
	static LinkedHashMap<String,Integer> copyDictionary = new LinkedHashMap<String,Integer>();
    
	private final static String[] hexSymbols = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F" };

    public final static int BITS_PER_HEX_DIGIT = 4;

    public static String toHexFromByte(final byte b)
    {
        byte leftSymbol = (byte)((b >>> BITS_PER_HEX_DIGIT) & 0x0f);
        byte rightSymbol = (byte)(b & 0x0f);

        return (hexSymbols[leftSymbol] + hexSymbols[rightSymbol]);
    }

    public static String toHexFromBytes(final byte[] bytes)
    {
        if(bytes == null || bytes.length == 0)
        {
            return (" ");
        }

        // there are 2 hex digits per byte
        StringBuilder hexBuffer = new StringBuilder(bytes.length * 2);

        // for each byte, convert it to hex and append it to the buffer
        for(int i = 0; i < bytes.length; i++)
        {
            hexBuffer.append(toHexFromByte(bytes[i]));
        }
       // hexBuffer.append(" ");
        return (hexBuffer.toString());
    }

	public static void updateMap( FileInputStream fis)
	{	
		LinkedHashMap<String,Integer> local_dict = new LinkedHashMap<String,Integer>();
		HashSet<String> set = new HashSet<String>();
        try
        {
          //  FileInputStream fis = new FileInputStream(new File(args[0]));
           // BufferedWriter fos = new BufferedWriter(new FileWriter(new File(args[1])));
            
            byte[] b1 = new byte[1];
            byte[] b2 = new byte[1];
            byte[] b3 = new byte[1];
            byte[] b4 = new byte[1];
			
            int value = 0;
            /*do
            {
                fos.write(toHexFromBytes(b1));
                fos.write(toHexFromBytes(b2));
                fos.write(toHexFromBytes(b3));
                fos.write(toHexFromBytes(b4));
                fos.write("\n");
                b1 = Arrays.copyOf(b2, 1);
                b2 = Arrays.copyOf(b3, 1);
                b3 = Arrays.copyOf(b4, 1);
                value = fis.read(b4);

            }while(value != -1);

            fos.flush();
            fos.close();
        }*/
		
			do
			{
				int count;
				String temp = "";
				temp += toHexFromBytes(b1);
				temp += toHexFromBytes(b2);
				temp += toHexFromBytes(b3);
				temp += toHexFromBytes(b4);
				if(local_dict.containsKey(temp))
				{
					count = local_dict.get(temp);
					//System.out.println(count);
					local_dict.put(temp,++count);
				}
				else
				{
					local_dict.put(temp,1);
				}
				//System.out.println(temp);
				b1 = Arrays.copyOf(b2, 1);
				b2 = Arrays.copyOf(b3, 1);
				b3 = Arrays.copyOf(b4, 1);
				value = fis.read(b4);
				
			}while(value != -1);
			
			Set<Map.Entry<String, Integer>> local_set = local_dict.entrySet();
			List<Map.Entry<String, Integer>> list = new ArrayList<Map.Entry<String, Integer>>(local_set);
			Collections.sort( list, new Comparator<Map.Entry<String, Integer>>()
			{
				public int compare( Map.Entry<String, Integer> o1, Map.Entry<String, Integer> o2 )
				{
					return (o2.getValue()).compareTo(o1.getValue());
				}
			} );
			
			int times = 0;
			int count = 0;
			for(Map.Entry<String, Integer> entry:list)
			{
				 if(times < 30)
				 {
					if(dictionary.containsKey(entry.getKey()))
					{
						count = dictionary.get(entry.getKey());
						//System.out.println(count);
						dictionary.put(entry.getKey(), count + entry.getValue());
						times++;
					}
					else
					{
						dictionary.put(entry.getKey(),entry.getValue());
						times++;
					}
				 }
				 else
					break;
			}
	
			
			/*for(String temp1:dictionary.keySet())
			{
				System.out.println(temp1+" "+dictionary.get(temp1));
			}*/
		
		}		
		catch(Exception e)
		{
			e.printStackTrace();
		}
	}	
	
	public static void printMap()
	{
		for(Map.Entry<String,Integer> m:dictionary.entrySet())
		{  
			System.out.println(m.getKey()+" "+m.getValue());  
		}  
	}

	public static void serialise()
	{
		try
		{
			FileOutputStream fileOut = new FileOutputStream("cout.ser");
			ObjectOutputStream out = new ObjectOutputStream(fileOut);
			out.writeObject(copyDictionary);
			out.close();
			fileOut.close();
		}
		catch (IOException e)
		{
			e.printStackTrace();
		}
	}

	public static void deserialise()
	{
      	try
      	{
        	FileInputStream fileIn = new FileInputStream("cout.ser");
        	ObjectInputStream in = new ObjectInputStream(fileIn);
	        dictionary = (LinkedHashMap<String, Integer>) in.readObject();
	        in.close();
	        fileIn.close();
      	}

      	catch(IOException i) 
      	{
        	i.printStackTrace();
      	}

      	catch(ClassNotFoundException c) 
      	{
	        c.printStackTrace();
      	}
	}

	public static void process(String filename)
	{
		try
		{
			FileInputStream fis1 = new FileInputStream(new File(filename));
			updateMap(fis1);
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
	}
	
	public void generate(String filename)
	{
		//Write a function that reads a hex from a file 
		//check if any of the hex feature is present in that file


	}

    public static void main(String[] args)
    {
		try
        {
        	/*
        	serialise();
            File folder = new File("mexe/");
            File[] listOfFiles = folder.listFiles();
            for (File file : listOfFiles) 
            {
    			if (file.isFile()) 
    			{
    				deserialise();
			        process("mexe/"+file.getName());
			        serialise();
    			}
			}
			printMap();
			*/

			deserialise();
			File folder = new File("mexe/");
            File[] listOfFiles = folder.listFiles();
            for (File file : listOfFiles) 
            {
    			if (file.isFile()) 
    			{
			        generate("mexe/"+file.getName());
    			}
			}

			/*
			Set<Map.Entry<String, Integer>> set = dictionary.entrySet();
			List<Map.Entry<String, Integer>> list = new ArrayList<Map.Entry<String, Integer>>(set);
			Collections.sort( list, new Comparator<Map.Entry<String, Integer>>()
			{
				public int compare( Map.Entry<String, Integer> o1, Map.Entry<String, Integer> o2 )
				{
					return (o2.getValue()).compareTo(o1.getValue());
				}
			} );

			for(Map.Entry<String, Integer> entry:list)
			{
				if(entry.getValue()>100)
					copyDictionary.put(entry.getKey(), entry.getValue());
			}
			serialise();
			*/

			printMap();
		}
		
		catch(Exception e)
        {
            e.printStackTrace();
        }
	}
}	