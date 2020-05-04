# get current dir
basename $(pwd)

# argument parsing
# https://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash

# get last item
dir=$(echo "abc/cd/e/fg/file" | rev | cut -d'/' -f-1 | rev)

# Substring removal
x="a1 b1 c2 d2"
echo ${x#*1} # b1 c2 d2
echo ${x##*1} # c2 d2
echo ${x%1*} # a1 b
echo ${x%%1*} # a
echo ${x/1/3} # a3 b1 c2 d2
echo ${x//1/3} # a3 b3 c2 d2
echo ${x//?1/z3} # z3 z3 c2 d2

# ' become string
# " use variable
'$1'
"$1"
# Hello ''
echo "Hello '$1'"

# if $var not exists, return error 
echo ${var?}

# check file exist
if [ ! -f dir_name/.gitignore ]; then fi

# check dir exist
if [ -d /kkbox-rdc-personal/kkaudio-cache/songs ]; then fi
[ -d foo ] || mkdir foo

# check null
if [ ! -z "$2" ]; then 
  echo "$2 is not empty"
fi

# string compare
if [[ "$1" == "f" ]]; then fi

# strstr(), string contains substring
if [[ "helloworld" =~ "hello" ]]; then
  echo "Contains"
fi
# another way, with regex
if [[ "helloworld" == *"hello"* ]]; then
  echo "Contains"
fi

# true/false
if [ "$var" == true ]; then fi

# success or fail
some_command
if [ $? -eq 0 ]; then
    echo OK
else
    echo FAIL
fi


# https://stackoverflow.com/questions/18709962/regex-matching-in-a-bash-if-statement
# if regex
regex="[0-9a-zA-Z ]"
if [[ $x =~ $regex ]]; then ...

# read stdout into array
my_array=()
while IFS= read -r line; do
    my_array+=( "$line" )
done < <( my_command )

# read lines
readarray song_ids < "$1"
for song_id in ${song_ids[@]}
do
    echo $song_id
done

for i in {1..10};do
  echo $i
done

END=5
for ((i=1;i<=END;i++)); do
  echo $i
done

# sorting

# sort by file size

# du : estimate file disk usage.
# -h : for human
# -a : all files

# sort : sort lines of text.
# -h : for human
du -ha | sort -h


# [2] of list
cut -d "/" -f 2
# [:2] of list
cut -d "/" -f -2

# find by size, and then remove heading ./
find . -size +1M | sed 's/^.\///g'

#####################
# File Name Editing #
#####################

# rename multiple files using regex
# -n is --no-act
rename -v -n 's/file_\d{1,3}/upl/' file_*.png


### SSH
# ssh remote command
ssh server_ip 'cd ~/Desktop; ls -a'

# mute all output
1>/dev/null 2>&1

### Port Forwarding

ssh [-nNT] -L ${localhost_port}:${server_IP}:${server_port} ${bridge_server_IP}
# 開jupyter notebook
ssh -nNT -L 8080:localhost:8888 ubuntu@192.168.200.159 
pgrep ssh | xargs kill

### Proxy Server
ssh [-nNT] -D ${port} ${server_IP}

### Use local public key as server public key
ssh -a -i path/to/public/key server_IP

# os level command like crontab, nohop should use absolute path
crontab
nohup  [command] > /dev/null 2>&1 &
nohup bash madmom.sh song_id.csv>bpm.out 2>/dev/null &

## SSH without password

ssh-keygen
ssh-copy-id -i ~/.ssh/id_rsa.pub remotehost_IP

pbcopy < ~/.ssh/id_rsa.pub

# in ~/.ssh/config
# to connect to server2, tunnel through server1
Host server2
ProxyCommand ssh server1 nc %h %p
# need to

# to connect to server3, tunnel through server2
Host server3
ProxyCommand ssh server2 nc %h %p


Network command
show all ethernet interface

    ifconfig

connect to the server insecurely 

    telnet host port

port scanner

    nmap IP
    # nmap yahoo.com
    # PORT    STATE SERVICE
    # 25/tcp  open  smtp
    # 80/tcp  open  http
    # 443/tcp open  https

network statistics 

    netstat

display routers between client and server

    traceroute IP

check network connectivity through icmp 

    ping IP


TCP/UDP connection
netcat/nc

    # start a listening server
    [server] netcat -l localhost 8888
    # connect to the server on port
    [client] netcat localhost 8888
    
    # udp
    [server] nc -u -l 10000  
    [client] nc -u server 10000 

collect packages going through the eth

    tcpdump -i ethN 



# http://wanggen.myweb.hinet.net/ach3/ach3.html?MywebPageId=201711496309082311#awk_basic

# row
awk {'print $1'}
# 'fast': 1223 -> extract 1223
awk "match(\$0,/'fast': [0-9]+/) {print substr(\$0,RSTART+8,RLENGTH-8)}"

# column 
grep

# search all files containing pattern
grep -rnw path/to/files/ -e 'util\/signal_source[._].*\"'


# stream editor
# https://edoras.sdsu.edu/doc/sed-oneliners.html
sed

# \1 is group one

# preview mode
# -n tell sed to not automatically print all lines. The the p at the end tell sed to print the lines that match your search
sed -n 's/util\(\/signal_source[._]\)/util\/signal_source\1/gp' fl_selection.*

# preview by changed lines only
# TODO multiple files
sed 's/\(DECLARE_STATEFUL_FN\)_1/\1/g' lstm/fl_sequence_balancer.cc | diff -y --suppress-common-lines lstm/fl_sequence_balancer.cc -

# -i in-place replacement
# TODO recursively
sed -i 's/util\(\/signal_source[._]\)/util\/signal_source\1/g' fl_selection.*

# rename files, -n is preview mode
rename -n 's/(\w+) - (\d{1})x(\d{2}).*$/S0$2E$3\.srt/' *.srt

# each argument
xargs

# multiple commands for each xargs
# -I means argument
a.txt xargs -I {} sh -c 'command1; command2; ...{}'

#calculate numbers
num=1
num=$(( $num + 1 ))

# variable length
var=testing
echo ${#var}


# shell arguments
# http://osr507doc.xinuos.com/en/OSUserG/_Passing_to_shell_script.html

# Double Quotes: 
# Although double quotes preserves literal value of each character within the quotes, 
# it has certain exceptions according to the bash manual. It will exclude $, `, \ and ! (when history expansion is enabled)

# Single Quotes: 
# Enclosing characters in single quotes (') preserves the literal value of each character within the quotes.

# Escape without Quotes: 
# Without quotes, shell split input into words, applies its quoting rules to each tokens and treats 
# the character next to \ with its literal value accordingly. Thus it considers !as ! instead of expanding history.


### ln
# i node
ls -i file_name


### kill running port
lsof -i tcp:3000
kill -9 PID

### count lines of code
find directory -name '*.py' | xargs wc -l

