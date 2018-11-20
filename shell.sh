# get current dir
basename $(pwd)

# argument parsing
# https://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash

# get last item
dir=$(echo "abc/cd/e/fg/file" | rev | cut -d'/' -f-1 | rev)

# ' become string
# " use variable
'$1'
"$1"
# Hello ''
echo "Hello '$1'"

# check file exist
if [ ! -f dir_name/.gitignore ]; then fi

# check dir exist
if [ -d /kkbox-rdc-personal/kkaudio-cache/songs ]; then fi
[ -d foo ] || mkdir foo

# check null
if [ -z "$2" ]; then fi

# string compare
if [[ "$1" == "f" ]]; then fi

# true/false
if [ "$var" == true ]; then fi

# read lines
readarray song_ids < "$1"
for song_id in ${song_ids[@]}
do
    echo $song_id
done

# [2] of list
cut -d "/" -f 2
# [:2] of list
cut -d "/" -f -2

# find by size, and then remove heading ./
find . -size +1M | sed 's/^.\///g'

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

# column 
grep

# row
awk {'print $1'}
# 'fast': 1223 -> extract 1223
awk "match(\$0,/'fast': [0-9]+/) {print substr(\$0,RSTART+8,RLENGTH-8)}"

# stream editor
sed

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


### ln
# i node
ls -i file_name


### kill running port
lsof -i tcp:3000
kill -9 PID

### count lines of code
find directory -name '*.py' | xargs wc -l



