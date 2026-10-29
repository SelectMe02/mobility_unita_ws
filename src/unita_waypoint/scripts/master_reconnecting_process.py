#!/usr/bin/env python3
"""Keep a debug process registered after the external ROS master restarts.

No rospy node is created here. The child owns the original node name/topics.
Only this debugger's private parameters are restored to a replacement master.
"""
import os
import signal
import subprocess
import sys
import threading
import xmlrpc.client


class TimeoutTransport(xmlrpc.client.Transport):
    def make_connection(self, host):
        connection = super().make_connection(host)
        connection.timeout = 1.0
        return connection


def node_name(args, environment):
    remaps = dict(arg.split(':=', 1) for arg in args if ':=' in arg)
    namespace = remaps.get('__ns', environment.get('ROS_NAMESPACE', '/'))
    return '/' + '/'.join(part for part in
        (namespace.strip('/'), remaps.get('__name', 'debug_view').strip('/')) if part)


class MasterRegistration:
    def __init__(self, name):
        self.name = name
        self.identity = None
        self.parameters = None

    def observe(self, master):
        code, _, pid = master.getPid(self.name)
        if code != 1:
            return False
        code, _, run_id = master.getParam(self.name, '/run_id')
        identity = (pid, run_id if code == 1 else None)
        if self.identity is None:
            code, _, values = master.getParam(self.name, self.name)
            if code == 1 and isinstance(values, dict):
                self.parameters = values
            self.identity = identity
            return False
        if identity == self.identity:
            return False
        # Commit identity only after successful restoration; failures retry.
        if self.parameters is not None:
            code, message, _ = master.setParam(self.name, self.name, self.parameters)
            if code != 1:
                raise RuntimeError(message)
        self.identity = identity
        return True


def stop_child(child):
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGINT)
        child.wait(timeout=3.)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
    except ProcessLookupError:
        pass


def main():
    if len(sys.argv) < 3:
        raise SystemExit('usage: master_reconnecting_process.py PACKAGE EXECUTABLE [arguments]')
    command = ['rosrun'] + sys.argv[1:]
    registration = MasterRegistration(node_name(sys.argv[3:], os.environ))
    master = xmlrpc.client.ServerProxy(os.environ.get('ROS_MASTER_URI', 'http://localhost:11311'),
                                      transport=TimeoutTransport())
    stopped = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.set())
    child = None
    try:
        while not stopped.is_set():
            try:
                changed = registration.observe(master)
                if changed:
                    print('ROS master restarted; reconnecting ' + registration.name, flush=True)
                    stop_child(child)
                    child = None
                if registration.identity is not None and child is None:
                    child = subprocess.Popen(command, start_new_session=True)
            except (OSError, xmlrpc.client.Error, RuntimeError) as error:
                # Retain the viewer while the master is temporarily absent.
                pass
            if child is not None and child.poll() is not None:
                # Respect an intentional RViz/window close or normal node exit.
                return child.returncode
            stopped.wait(.5)
    finally:
        stop_child(child)
    return 0


if __name__ == '__main__':
    sys.exit(main())
