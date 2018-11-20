import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class usefullibrary{
	public static void main(String[] args) {
	}
	public void tools(){
        /* ARRAY */    
		int[] x = {1,2,3};
		int[] y = new int[10];//call by reference
        int[] z = new int[] {
            1, 2, 3
        };
		int[][] two = {{1,2},{1,4}};

		int[][] 2d = new int[a.length][];
		y.length == 3;
		print Arrays.toString(y);
		Arrays.fill(ids, -1);
		Arrays.sort(y);//primitive
		Arrays.sort(people, (a,b) -> a[0] == b[0] ? a[1]-b[1] : a[0]-b[0]); //2d primitive
        copyOfX = x.clone()
        Arrays.copyOfRange(x, x.length-2, x.length)
        Arrays.asList(theArray).indexOf(o)

        /* STRING */
		StringBuilder sb = new StringBuilder(); //call by reference
		sb.append("adsfsafsd");
		sb.insert(0, "sdssds");
		sb.charAt(0);
		sb.indexOf(str);//not found -1
		sb.length();
		sb.toString();
        sb.substring();
		sb.setCharAt(0, 's'); //void
        sb.replace(int start, int end, String str) // StringBuilder
		sb.reverse().toString()
		sb.delete(0,1);

        String.join(" ", "one", "two", "three");
        String.join(" ", List<String>);
        "abc".equals("abc");//use equals in Object
		String s = "a" + "bcbc"; //immutable, call by value
		Strin x = new String(s.toCharArray());
		s.charAt(0);
		s.length();
		s.contains("ab");
        //check white space
        s.contains(" ");
        //check null/empty string
        if (string.trim().length() > 0)
		s.indexOf("bc");
		"\t\t\taaa".lastIndexOf("\t") // 2, -1 if none
        // need to handle array lenght == 0
		s.split("\\s+");
        "1.1.2".split("\\."); // split take regex
		"a".split(",") -> "a"
		"".split(",") -> size 1 array  
		",".split(",") -> size 0 array
		s.toLowerCase();
		s.toUpperCase();
		s.trim();//s = "   sf ", s.trim() == "sf"
		s.substring(int beginIndex, int endIndex)
		for (char ch : s.toCharArray())
        String.valueOf(s.toCharArray()); // convert char array to string

        /* String convertion */
        // char array to string
        char[] chars = {'a', 'b', 'c'};
        String s = new String(chars);

        // int to string
		String s = Integer.toString(int i);
		String s = i + "":

        // string to int
		int a = Integer.parseInt(String s);
        // char to int or char
        int a = '9' - '0';
		char ch = (char) 'B'-'A' + 'a'; //'b'

        // These two have the same value
        new String("test").equals("test") // --> true 
        // ... but they are not the same object
        new String("test") == "test" // --> false 
        // ... neither are these
        new String("test") == new String("test") // --> false 
        // ... but these are because literals are interned by 
        // the compiler and thus refer to the same object
        "test" == "test" // --> true 
        // ... string literals are concatenated by the compiler
        // and the results are interned.
        "test" == "te" + "st" // --> true

        /* LIST */
		List a;
		a.size();
		List<int[]> list = new ArrayList<>();
		List<List<Integer>> l = new ArrayList<>();
		l.sort((a, b) -> a.get(0) == b.get(0) ? b.get(1) - a.get(1) : a.get(0) - b.get(0));
		l.toArray(new int[people.length][]);
        list.get(idx);
        list.set(idx, val);
        list.add(29);
        list.add(29, idx);
        list.remove(list.size() - 1);
        list.contains(val);
        new ArrayList(input.subList(0, input.size()/2))

        /* STACK */
		Stack<E> stack = new Stack<>();
		stack.push(e);
		stack.pop();// exception if empty
		stack.peek();
		stack.empty();

        /* QUEUE */
		Queue<E> queue = new LinkedList<>();
		queue.offer(E);
		queue.poll(); //null if empty
		queue.peek();
		queue.isEmpty();
        //min heap, poll min, offer bottom, maintain kth largest
		Queue<E> minHeap = new PriorityQueue<>();
        //max heap, poll max, offer bottom, maintain kth smallest
		Queue<E> maxHeap = new PriorityQueue<>((a,b) -> b - a);

        /* MAP */
		Map<K, V> map = new HashMap<>();
		Map<K, V> map = new TreeMap<>(); //AVL BST tree, iterate sorted key
		Map<K, V> map = new LinkedHashMap<>(); //iterate inserted order
		map.size(); 
        map.isEmpty();
		map.get(k); //null if not found
		map.put(k, v);
		map.remove(k);
        //ascending iteration
		for (K k : map.keySet());
		for (V v : map.values());
		map.containsKey(k);
		map.containsValue(v);
        map.getOrDefault(Object key, V defaultValue);
        int[] map = new int[256];

        /* SET */
        Set<E> set = new HashSet<>();
        set.add(e);
        set.contains(e);
        set.isEmpty();
        set.remove(e);
        set.size();
        for (E k: set)
        set.toArray();


        /* Iterator */
        Iterator<E> iterator = set.iterator();
        for (Iterator<E> iterator = set.iterator(); iterator.hasNext();) {
            E e = iterator.next();
             //Iterator.remove() is the only safe way to modify a collection during iteration
            if (e has conditions) iterator.remove();
        }

        /* JAVA */
		obj1 == obj2; //reference equality
		obj1.equals(obj2); //value equality
		String ab = new String("ab");
		"ab" == "ab";//true
		ab == "ab";//false
		"ab".equals("ab");//true
		Object a;
		System.out.println(a);//compile error
		Object b = null;
		System.out.println(b);//null
		int c;
		System.out.println(c);//0

        /* MATH */
		if (overflow = Integer.MAX_VALUE)
		if (underflow = Integer.MIN_VALUE)
		Integer.signum(-123);
		Math.max(a, b);
		Math.abs(a);
		Math.pow(a, 2);
		Math.sqrt(a);
        Math.log(a);
        Math.ceil(a);
        Math.floor(a);

        /* REGEX */
        import java.util.regex.Matcher;
        import java.util.regex.Pattern;

        String str = "word regular expression";
        Pattern pattern = Pattern.compile("\\w+\\b");
        Matcher m = pattern.matcher(str);
        while (m.find()) System.out.println(m.group());

        /* STREAM */
        // char stream
        int sum = Integer.toString(1234).chars()
        .map(i -> (int) Math.pow(i - '0', 2))
        .reduce(0, (a, b) -> a + b);

        int[] chars = new int[26];
        "abbcd".chars().forEach(e -> chars[e - 'a']++);

        // int stream
        int[] nums = {1, 2, 3, 4};
        Arrays.stream(nums).min().getAsInt();

        // I/O
        Scanner in = new Scanner(System.in);
		BufferedWriter bw = null;
		FileWriter fw = null;
		try{
			fw = new FileWriter("output2.txt");
			bw = new BufferedWriter(fw);

            while(in.hasNext()) {
                int a = in.nextInt();
                int b = in.nextInt();
                System.out.println(a + b);
                bw.write("Case #" + a + ": " + b +"\n");
            }

			bw.close();
			fw.close();
		} catch (IOException e) {
			// do something
		}
        if (A == null || A.length == 0) {
            return -1;
        }
	}
}
