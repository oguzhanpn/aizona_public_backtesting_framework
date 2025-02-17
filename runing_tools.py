import paramiko


def run_strategy_on_server(username, password, script_path, file_name):

    hostname = "213.142.148.171"

    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname, username=username, password=password)
        sftp = ssh.open_sftp()

        local_file = script_path
        remote_path = file_name
        sftp.put(local_file, remote_path)

        print(f"File uploaded successfully to remote server you can proceed to discord channel for the remaining logs")

        sftp.close()
        ssh.close()

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        print(f"Error: {e}")

