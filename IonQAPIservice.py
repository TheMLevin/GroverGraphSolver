#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  Class 'IonQAPIservice': use IonQ's REST API to submit and retrieve jobs. 
#
#  for detailed information 
#     use: <instance> = IonQAPIservice("ionq_api_token").help()
##########################################################################

import os, io, time, math
from datetime import datetime, timedelta
import string, re
import json
import requests


class IonQAPIservice:

   def help(self, on_method="LIST"):
      """
help(on_method) - print help text 
---------------
  input: on_method for which documentation is requested or  
         "ALL" for all methods or "LIST" for simple list (default: on_method="LIST")
      """
      helptext = {}
      helptext["class"] = """
-----------------------------------------------------------------------
Class 'IonQAPIservice': use IonQ's REST API to submit and retrieve jobs. 
-----------------------------------------------------------------------
The IonQ API offers noise models for the 'Harmony' (11 qubits) and 'Aria-1' (25 qubits) hardware.
To use these options, and also to run circuits on IonQ's Aria-1 quantum computer, the circuit is 
  translated into JSON format and then submitted to IonQ's servers.
      """
      helptext["class2"] = """
instantation: x = IonQAPIservice("api_token")
          or: x = IonQAPIservice()  if environment variable IONQ_API_TOKEN or IONQ_API_KEY or QISKIT_IONQ_API_TOKEN is defined
       optional 2nd argument: jobid_file_base (basename for saving jobIDs (UID strings) and submission time - only for jobs submitted to a QPU target, not simulation jobs),
         e.g. x = IonQAPIservice(jobid_file_base="myjobs")  will create or append to file "myjobs.txt" 
                    in the same directory lines like: "submission_time : jobid_hash_string"
                    (the output file can also be defined by optional argument 'save_file=<filename>' 
                      in methods submit_job() or submit_multiple_jobs() )
      """
      helptext["header1"] = """
=== Methods for job submission and retrieval of results:
      """
      helptext["submit_job"] = """ 
submit_job(circuit_or_file, wait_minutes, verbose, **kwargs)
------------------------------------------------------------
  returns: dictionary with job_id, status, and input information as returned by IonQ;
        if the job is completed, the dictionary also contains the results ('counts' and 'probabilities') 
           under key 'results' like {'results': {'counts': {...}, 'probabilities': {...}}};
    on error returns: dictionary {'error': error_message, 'data': submitted circuit_or_file}

  circuit_or_file: quantum circuit (qiskit.QuantumCircuit or cirq.Circuit) or OPENQASM circuit (string or list) or JSON format (file or string or dict);

  wait_minutes: after submission the function calls retrieve_job to get the job status (usually returning 'status':'submitted' or 'ready');
    if wait_minutes>0, the returned dictionary also contains a 'results' key like {'results': {'counts': {...}, 'probabilities': {...}}};

  verbose: verbosity (True or False), default: True 

  **kwargs: catch all for additional parameters for the circuit before submission, e.g.
    shots=...: number of repetitions; default: shots=1024;
    noise_model="...": noise model to be used for simulator; default: noise_model="ideal"; other options: "harmony", "aria-1";
    sampler_seed=...: seed for random number generator used in simulator; default: not provided);
    noise={"model": "...", "seed": ...}: same as the previous two options together;
    target="...": provide backend 'target' (default: target="simulator"; other options: "qpu.harmony" (same as: "qpu"), "qpu.aria-1");
    backend=... (string or pointer to IonQBackend class): similar to option target but for backend=[pointer to IonQBackend instantation] together with circuit_or_file=[pointer to QuantumCircuit instantation] the qiskit_ionq API is used to translate the circuit for the given IonQBackend class;
      """
      helptext["retrieve_job"] = """
retrieve_job(jobid, wait_minutes, verbose)
------------------------------------------
  returns: dictionary with job_id, status and input information as returned by IonQ, if the job is completed, 
           the returned results are provided under dictionary key {'results: {'counts": {...}, 'probabilities': {...}}};
    on error returns dictionary with entry {'error': error message}

  jobid: a UID (hash-string) or dictionary with entry 'id': 'UID'; 
         if no jobid is provided, the last used or returned jobid is used (e.g. the jobid returned when submitting a job);

  wait_minutes: wait for up to wait_minutes=N minutes for the job to complete (status is checked every 30 seconds), default is N=0;

  verbose: verbosity (True or False), default: True 
      """
      helptext["cancel_job"] = """
cancel_job(jobid, verbose) 
--------------------------
  returns: dictionary with entries {'id': jobid, 'status': 'canceled'},
    on error returns dictionary with entry {'error': error message}

  jobid: jobID or dict containing jobID to cancel (only jobs with status 'submitted' or 'ready' can be canceled 
         (if no jobid is provided, the last jobid is used).

  verbose: verbosity (True or False), default: True 
      """
      helptext["submit_multiple_jobs"] = """
submit_multiple_jobs(list_of_circuits, wait_minutes, verbose, **kwargs)
-----------------------------------------------------------------------
  returns dictionary with entry {'circuit_name': out_dict}, where out_dict is the returned dict from submit_job
    on error returns dictionary with entry {'error': error message}

  list_of_circuits list of dicts or files defining circuits or list of QuantumCircuits;

  wait_minutes, verbose, **kwargs: see function submit_jobs;
    here: name=... specifies a name base (circuit number is added to the 'name');
    alternatively provide a list of circuit names with names=[...].

  verbose: verbosity (True or False), default: True 

  **kwargs: see note under 'submit_job'
      """
      helptext["retrieve_multiple_jobs"] = """
retrieve_multiple_jobs(list_dict_file, wait_minutes, verbose)
-------------------------------------------------------------
  returns dictionary with entry {'circuit_name': out_dict}, where out_dict is the returned dict from retrieve_job
    on error returns dictionary with entry {'error': error message}

  list_dict_file list or dict or file containing jobIDs (one per line; with or without circuit_name;

  wait_minutes, verbose: see function retrieve_job.

  verbose: verbosity (True or False), default: True 
      """
      helptext["cancel_multiple_jobs"] = """
cancel_multiple_jobs(list_dict_file, verbose)
---------------------------------------------
  returns dictionary {'ids': list_of_cancelled_jobIDs};
    on error returns dictionary with entry {'error': error message};

  list_dict_file list or dict or file containing jobIDs (one per line; with or without circuit_name).

  verbose: verbosity (True or False), default: True 
      """
      helptext["extract_probabilities"] = """
extract_probabilities(jobdict)
------------------------------
  returns probabilities for all circuit states with non-zero probabilities ;
    on error prints error message and returns None;

  jobdict: dictionary returned by IonQ API or dictionary of form {'circuit_name': jobdict_from_Ionq_API}
    (best to use the returned dictionary of function retrieve_job or retrieve_multiple_jobs
    or of function submit_job, submit_multiple_jobs, preferably with input wait_minutes>0)
    if the input 'jobdict' is a valid JobID (UID string) or a list of jobID strings, the function will call retrieve_job first.
    if no jobid is provided, the last jobid is used (e.g. the jobid returned when submitting a job 
      or the jobid used to retrieve results)
      """
      helptext["extract_counts"] = """
extract_counts(jobdict)
-----------------------
  returns measured or simulated counts for all circuit states with non-zero counts;
    on error prints error message and returns None;

  jobdict: dictionary returned by IonQ API or dictionary of form {'circuit_name': jobdict_from_Ionq_API}
    (best to use the returned dictionary of function retrieve_job or retrieve_multiple_jobs
    or returned dict of function submit_job, submit_multiple_jobs, preferably with input wait_minutes>0)
    if the input 'jobdict' is a valid JobID (UID string) or a list of jobID strings, the function will call retrieve_job first.
    if no jobid is provided, the last jobid is used (e.g. the jobid returned when submitting a job 
      or the jobid used to retrieve results)
      """
      helptext["header2"] = """
===  Helper methods: 
      """
      helptext["translate_qasm"] = """
translate_qasm(qasm_qc_list, verbose)
-------------------------------------
   returns: JSON dict (string) with "body" dictionary for IonQ API, 
    on error returns: None

   qasm_qc_list: list or string with OPENQASM commands (starting with OPENQASM version; include ...;)
    a string with QASM circuit can easily be obtained by the method: quantumCircuit.qasm() 

  verbose: verbosity (True or False), default: True 
      """
      helptext["validate_circuit"] = """
validate_circuit(circuit) 
-------------------------
  returns: dictionary {'circuit': validated and corrected circuit, 'updated_entries': list of modified gate entries} 
    on error returns: dictionary {'error': error_message} 

  circuit: list of operations as being submitted to IonQ (operations are dicts of form {'gate': ..., 'target' or 'targets': ..., 'control' or 'controls': ...}). 
      """
      helptext["validate_jobid_hash"] = """
validate_jobid_hash(jobid)
--------------------------
  The routine checks whether the provided 'jobid' is a UID, i.e. hash string containing 5 hex numbers of correct lengths;
  returns: dictionary {'id': jobid} 
    on error returns: dictionary {'error': error_message} 

  jobid: UID (hash-string) or dictionary containing entry {'id': jobid}. 
      """
      helptext["get_jobids_from_input"] = """
get_jobids_from_input(list_dict_file, name_base)
------------------------------------------------
  The routine tries to retrieve jobIDs from a list or dictionary or file; 
  returns dictionary of form {'circuit_name': jobID }; 
    on error returns: dictionary {'error': error_message} 

  list_dict_file: Input list of jobIDs or dictionary of form {'ids': list_of_jobIDs} or 
    of form {'circuit_name': jobID} or file containing one JobID per line or containing one circuit_name and jobID per line; 

  name_base: if the circuit names are not provided in the input, the name is given by 'name_baseN' 
    where N is the entry number (element# of list or dict or line# of file), default name_base="circuit_". 
      """
      helptext["get_waittime"] = """
get_waittime(backend, calib_data) 
---------------------------------
   - print current wait time (average queue time) for provided backend
   input arguments:
    backend: for which IonQ API backend to print: "simulator" or "qpu.harmony" or "qpu.aria-1" ...
       backend="ALL": print wait time, status, and access information for all existing backends
       backend="ACCESS" (default): print wait time and status for all backends, to which we have access
    calib_data: flag on whether also to print calibration/characterization results (default: False)
      """
      helptext["help"] = """
help(on_method) - print help text 
--------------
  input: on_method for which documentation is requested  (e.g. help("submit_job") )
         or "ALL" for all methods or "LIST" for simple list  (default: on_method="LIST")
      """

      all_methods = ["class", "class2", "header1", "submit_job", "retrieve_job", "cancel_job", 
                     "submit_multiple_jobs", "retrieve_multiple_jobs", "cancel_multiple_jobs",
                     "extract_probabilities", "extract_counts",
                     "header2", "translate_qasm", "validate_circuit", "validate_jobid_hash", 
		     "get_jobids_from_input", "get_waittime", "help"]

      if on_method in ["ALL", "LIST"]:
         print()
         for m in all_methods:
           if m in ["class", "header1", "header2", "help"]:
              print(helptext[m])
           elif on_method == "LIST":
              if m != "class2":
                 print(helptext[m].split("\n")[1])
           else:
              print(helptext[m])

      else:
         if on_method in all_methods:
            print(helptext[on_method])
         else:
            print("*** This method does not exist, use 'help()' or 'help(method)' where")
            print("***   'method' is one of:", end=" ")
            for m in all_methods:
               if m != "class2" and not m.startswith("header"): 
                  print(m, end=", ")
            print("\n")

######################################################################################


   def __init__(self, ionq_token=None, jobid_file_base='qpu_job'):

      if ionq_token is None:
         ionq_token = os.getenv("IONQ_API_KEY") or os.getenv("IONQ_API_TOKEN") or os.getenv("QISKIT_IONQ_API_TOKEN")

      if ionq_token is None:
         print('ERROR: IonQ API token not provided; use: IonQAPIservice("ionq_token")')
         return

      requests.packages.urllib3.disable_warnings()
     
      # check whether the provided IonQ API token has access to QPU hardware
  
      header = {'Authorization': f'apiKey {ionq_token}',}

      response = requests.get(f'https://api.ionq.co/v0.3/backends', headers=header)

      if response.ok:

         self.api_header = header
         self.has_access = {}
         data = response.json()

         # report status of IonQ backends:
         for backend in data:
            self.has_access.update({ backend["backend"]:backend["has_access"] })

         self.get_waittime()
            
      else:

         errtxt = f"{response}"
         if response.status_code == 400:
            errtxt += f": Bad request"

         elif response.status_code == 401:
            errtxt += ": Authentication failed"

         print('ERROR accessing IonQ Server using provided IonQ API token "'+ionq_token+'":', errtxt)

         self.api_header = {}

      self.__name__ = "IonQAPIservice"
      self.__version__ = 0.1
      self.last_jobid_dict = {'id': 'not set'}

      # local file to save jobIDs (UIDs=hash-strings) for jobs submitted to IonQ hardware 
      # (can be used later to retrieve results)
      if jobid_file_base is None:
         self.jobid_file_base = ""
      elif isinstance(jobid_file_base, bool):
         self.jobid_file_base = "qpu_jobs" if jobid_file_base == True else ""
      else:
         jobid_file_base = str(jobid_file_base)
         if len(jobid_file_base) > 4 and jobid_file_base.find(".", -5) >= 0:
            self.jobid_file_base = jobid_file_base
         else:
            self.jobid_file_base = jobid_file_base + ".txt"
      # note: if jobid_file_name=None or jobid_file_name='', then no file is created 
      #         (unless overridden by argument "save_file")

      # check whether qiskit and/or cirq are installed
      self.qiskit_installed = 2
      try:
         import qiskit
      except ImportError:
         self.qiskit_installed = 0
      if self.qiskit_installed > 0:
         try:
            import qiskit_ionq
         except ImportError:
            self.qiskit_installed = 1

      self.cirq_installed = 2
      try:
         import cirq
      except ImportError:
         self.cirq_installed = 0
      if self.cirq_installed > 0:
         try:
            import cirq_ionq
         except ImportError:
            self.cirq_installed = 1

      # rename some gates as IonQ is using different notations for some gates
      ### Gate list from IonQ API (https://docs.ionq.com/#section/Job-Inputs/Supported-Gates)
      ## Gate Description          Gate  Description
      ## x    Pauli X gate         not   alias for Pauli-X gate
      ## y    Pauli Y gate         cnot  alias controlled Pauli-X gate
      ## z    Pauli Z gate         v     Square root of not gate  (=sx=sqrt(not))
      ## h    Hadamard gate        vi    Conjugate transpose of square-root-of-not gate (=sxdg)
      ## s    S gate               si    Conjugate transpose of S gate
      ## t    T gate               ti    Conjugate transpose of T gate
      ## rx   X-axis rotation      xx    Ising XX gate: e^(-iθ X⊗X /2)
      ## ry   Y-axis rotation      yy    Ising YY gate: e^(-iθ Y⊗Y /2)
      ## rz   Z-axis rotation      zz    Ising ZZ gate: e^(-iθ Z⊗Z /2)
      ## swap Swaps two qubits

      # translation dict of form: 'qiskit_gate': 'ionq_gate'
      self.gates_1q = {'h':'h', 'i':'id', 'id':'id', 'p':'z', 'rx':'rx', 'ry':'ry', 'rz':'rz', 'not':'x',
            's':'s', 'sdg':'si', 'sx':'v', 'sxdg':'vi', 't':'t', 'tdg':'ti', 'x':'x', 'y':'y', 'z':'z'}
      self.gates_2q_ctl = {'ch':'h', 'cnot':'x', 'cp':'z', 'crx':'rx', 'cry':'ry', 'crz':'rz',  
                'cx':'x', 'cy':'y', 'cz':'z', 'csx':'v', 'cv':'v', 'csxdg':'vi', 'cvi':'vi'} 
      self.gates_2q_noctl = {'rxx':'xx', 'ryy':'yy', 'rzz':'zz', 'swap':'swap'}
      self.gates_mq = {'mcp':'z', 'mcphase':'z', 'mct':'t', 'ccx':'x', 'c3x':'x', 'c4x':'x', 
            'mcx':'x', 'mcx_gray':'x', 'toffoli':'x'}

      # 'special' gates, which are not provided by the IonQ API, are replaced by an equivalent sequence:
      #  format dict:  "gate_name" : [list_of_target_qubits, list_of_control_qubits, list_of_parameters,
      #            dict of gates and tuple with list of target, control, parameters used for this gate
      #   e.g. sswap(a,b)=sqrt(swap(a,b))=cnot(b,a)*csx(a,b)*cnot(b,a)
      #        iswap(a,b) = XX(pi)+YY(pi) = S(a)*S(b)*H(0)*CX(b,a)*CX(a,b)*H(b)
      #        siswap(a,b)=sqrt(iswap(a,b))
      self.gates_special = {'sswap': [[0,1],[],[], [('x',[1],[0],[]), ('v',[0],[1],[]), ('x',[1],[0],[]) ]],
                'iswap': [[0,1],[],[], [('s',[0],[],[]),('s',[1],[],[]),('h',[0],[],[]),('x',[1],[0],[]),
                                         ('x',[0],[1],[]),('h',[1],[],[]) ]],
                'siswap': [[0,1],[],[], [('v',[0],[],[]),('v',[1],[],[]),('rz',[0],[],[math.pi/2]),('x',[1],[0],[]),
                                         ('v',[0],[],[]),('v',[1],[],[]),('rz',[0],[],[7*math.pi/2]),
                                         ('rz',[1],[],[7*math.pi/2]),('v',[0],[],[]),('rz',[0],[],[math.pi/2]),
                                         ('x',[1],[0],[]),('v',[0],[],[]) ]],
      }


   def set_jobid_dict(self, jobid_dict=None):
      if jobid_dict is None:
        print("jobid_dict not specificed!  Usage:  .set_jobid_dict(jobid_dict)")
        return None

      if isinstance(jobid_dict, str):
         mydict = self.validate_jobid_hash(jobid_dict)

      elif isinstance(jobid_dict, dict):
         if "id" not in jobid_dict.keys():
            print("dict does not contain entry 'id'")
            return None
         mydict = self.validate_jobid_hash(jobid_dict["id"])

      elif isinstance(jobid_dict, list):
         mydict = self.validate_jobid_hash(jobid_dict[0])

      else:
         mydict ={'error': f"input '{jobid_dict}' does not contain a JobID hash string"}

      if "error" in mydict.keys():
          print(f"ERROR: {mydict['error']}")
          return None

      self.last_jobid_dict = jobid_dict if isinstance(jobid_dict, dict) else mydict 
      return self

   ####################################################
   #helper methods: 

   # to get an OPENQASM circuit, do: # qasm_string = qiskit_circuit.qasm()

   def translate_qasm(self, qc_list=None, verbose=False):
      """
   translate_qasm(qasm_qc_list, verbose)   
     returns: JSON dict (string) with "body" dictionary for IonQ API, 
       on error returns: None   
     input arguments:
      qasm_qc_list: list or string with OPENQASM commands (starting with OPENQASM version; include ...;)
        Note: a string with QASM circuit can easily be obtained by the method: quantumCircuit.qasm() 
      verbose: verbosity (True or False), default: True 
   """
      if qc_list is None: return None
      if verbose: print(qc_list, type(qc_list))

      if isinstance(qc_list, str):
         qc_list = qc_list.split(";")

      if qc_list[-1].strip() == '': 
         qc_list.pop(-1)

      qc_list = [re.sub('[;\r\t\n]','', j) for j in qc_list]
      qc_list = [j.strip() for j in qc_list if len(j.strip())>0 and not j.startswith('//')]
      if verbose: print(qc_list, type(qc_list))    

      #first entries in the list: 'OPENQASM 2.0','include "qelib1.inc"'
      str0 = qc_list.pop(0) ; str1 = qc_list.pop(1)

      if not str0.startswith("OPENQASM") or not str1.startswith("include"):
         if verbose: print('Header: "OPENQASM 2.0","include ..." missing')
         return None

      circ = [] ; qname = {}

      for j in range(len(qc_list)):

         vals = qc_list[j].split(" ")

         if vals[0] == "qreg":

            qlist = vals[1].split(",")

            if len(qlist) == 1:
               name,num = qlist[0].split("[")
               nqbits = int(num[:-1])
               qname = { name+'['+str(j)+']':j for j in range(nqbits)}
            else:
               nqbits = len(qlist)
               qname = { qlist[j]:j for j in range(qlist)}
            if verbose: print('qubits:',nqbits,qname)
            break

      if len(qname) > 0:

         for j in range(len(qc_list)):

            vals = qc_list[j].split(" ")

            if vals[0] in self.gates_1q.keys():
               circ.append({ "gate": f"{self.gates_1q[vals[0]]}", "target": qname[vals[1]] } )

            else:

               qval = vals[1].split(",")

               if qval[-1] in qname.keys():
                  qtgt = [ qname[qval[-1]] ]
                  qctl = [qname[qval[j]] for j in range(len(qval)-1)]

                  if vals[0] in self.gates_2q_noctl.keys():
                     circ.append({ "gate": f"{self.gates_2q_noctl[vals[0]]}", 
                                  "targets": qtgt.append(qname[qval[0]]) })

                  elif vals[0] in self.gates_2q_ctl.keys():   
                     circ.append({ "gate": f"{self.gates_2q_ctl[vals[0]]}", 
                                     "targets": qtgt, "controls": qctl })

                  elif vals[0] in self.gates_mq.keys():
                     circ.append({ "gate": f"{self.gates_mq[vals[0]]}", 
                                     "targets": qtgt, "controls": qctl })

      return json.dumps({"body": { "gateset": "qis", "qubits": len(qname), "circuit": circ } })

####################

   def validate_circuit(self, circuit):
      """
   validate_circuit(circuit) 
      returns: dictionary with entries: {'circuit': validated and corrected circuit, 'updated_entries': list of modified gate entries} 
       on error returns: dictionary {'error': error_message} 
      input arguments:
       circuit: list of operations as being submitted to IonQ (operations are dicts of form {'gate': ..., 'target' or 'targets': ..., 'control' or 'controls': ...}). 
      """

      valid_circ = []
      updates = []

      if not isinstance(circuit, list):
         return {"error": "'circuit' is not a list of entries (gates & target & control qubits)"}

      for j,entry in enumerate(circuit):

         if not isinstance(entry, dict):
            return {"error": f"'circuit' entry {j} is not dictionary"} 

         if "gate" not in entry.keys():
            continue

         mygate = entry["gate"]
         flg_update = False

         if "target" in entry.keys():
            tgt_vals = entry["target"]

            if isinstance(tgt_vals, list):
               flg_update = True
            else:
               tgt_vals = [tgt_vals]

         elif "targets" in entry.keys():
            tgt_vals = entry["targets"]

            if not isinstance(tgt_vals, list):
               flg_update = True
               tgt_vals = [tgt_vals]

         else:
            return {"error": f"'circuit' entry {j}: Gate '{mygate}' has no 'target'"}

         if "control" in entry.keys():
            ctl_vals = entry["control"]

            if isinstance(ctl_vals, list):
               flg_update = True
            else:
               ctl_vals = [ctl_vals]

         elif "controls" in entry.keys():
            ctl_vals = entry["controls"]

            if not isinstance(ctl_vals, list):
               flg_update = True
               ctl_vals = [ctl_vals]
         else:
            ctl_vals = []

         if "parameters" in entry.keys():
            parvals = entry["parameters"]

            if not isinstance(parvals, list):
               parvals = [parvals]
               flg_update = True

         elif "parameter" in entry.keys():
            parvals = [entry["parameter"]]
            
         if mygate in self.gates_special:

            qtgt = self.gates_special[mygate][0]
            qctl = self.gates_special[mygate][1]
            qpar = self.gates_special[mygate][2]

            if len(qtgt) != len(tgt_vals) or len(qctl) != len(ctl_vals):
               return{"error": 
                 f"'circuit' entry {j}: special gate {mygate} has wrong number of 'targets' and/or 'controls'"}

            qbits = tgt_vals + ctl_vals
            qgates = self.gates_special[mygate][3]

            for g,t,c,p in qgates:

               myt = [qbits[i] for i in t]
               circ = {"gate": f"{g}", "targets": myt}

               if len(c) > 0:
                  myc = [qbits[i] for i in c]
                  circ.update({"controls": myc})

               if len(p) > 0:
                  circ.update({"parameters": p})

               valid_circ.append(circ)

            updates.append(j)
            continue

         if len(tgt_vals) > 1:

            if mygate not in self.gates_2q_noctl.values():
               if mygate in self.gates_2q_noctl.keys():
                  mygate = self.gates_2q_noctl[mygate]
                  flg_update = True
               else:
                  return {"error": 
                        f"'circuit' entry {j}: gate '{mygate}' is not a valid gate with 2 targets"}

         elif len(ctl_vals) > 1:

            if mygate not in self.gates_mq.values():
               if mygate in self.gates_mq.keys():
                  mygate = self.gates_mq[mygate]
                  flg_update = True
               else: 
                  return {"error": 
                        f"'circuit' entry {j}: gate '{mygate}' is not a valid multi-qubit gate"}

         elif len(ctl_vals) == 1:

            if mygate not in self.gates_2q_ctl.values():

               if mygate in self.gates_2q_ctl.keys():
                  mygate = self.gates_2q_ctl[mygate]
                  flg_update = True

               elif mygate in self.gates_2q_noctl.values() or mygate in self.gates_2q_noctl.keys():
                  if mygate not in self.gates_2q_noctl.values():
                     mygate = self.gates_2q_noctl[mygate]
                  tgt_vals.extend(ctl_vals)
                  ctl_vals = []
                  flg_update = True

               else:
                  return {"error": 
                    f"'circuit' entry {j}: gate '{mygate}' does not have 'target' and 'control' qubits"}

         else:

            if mygate not in self.gates_1q.values():
               if mygate in self.gates_1q.keys():
                  mygate = self.gates_1q[mygate]
                  flg_update = True
               else:
                  return {"error":
                        f"'circuit' entry {j}: gate '{mygate}' is not a one-qubit gate"}

         if flg_update:
            updates.append(j)

            circ = {"gate": mygate, "targets": tgt_vals}
            if len(ctl_vals)>0:
               circ.update({"controls": ctl_vals})
            valid_circ.append(circ)

         else:
            valid_circ.append(entry)            

      if len(valid_circ) == 0:
         return {"error": "'circuit' contains no Gates"}

      return {'circuit': valid_circ, 'updated_entries': updates}

#######################

   def validate_jobid_hash(self,jobid):
      """
validate_jobid_hash(jobid)
    The routine checks whether the provided 'jobid' is a UID, i.e. hash string containing 5 hex numbers of correct lengths;
  returns: dictionary {'id': jobid} 
    on error returns: dictionary {'error': error_message} 
  input arguments:  
    jobid: UID (hash-string) or dictionary containing entry {'id': jobid}. 
      """

      if isinstance(jobid, dict):
         if "id" not in jobid.keys():
            return {"error" : f"{jobid} does not have key 'id'"}
         id = jobid["id"]

      elif isinstance(jobid, str):
         id = jobid.strip()

      else:
         return {"error": f"{jobid} is not a string"}

      if len(id)!=36 or id.count("-") != 4:
         return {"error": f"{id} cannot be a valid IonQ JobID"}

      vals = id.split("-")
      hashlen = [8, 4, 4, 4, 12]

      for n,val in zip(hashlen,vals):

         if len(val) != n:
            return {"error": f"{id} cannot be a valid IonQ JobID (hash of wrong length: {val})"}

         for ch in val:
            if ch not in [*'1234567890abcdef']:
               return {"error": f"{id} cannot be a valid IonQ JobID (hash with wrong hex char: {val})"}

      return {"id": id}

#########################

   def get_jobids_from_input(self, list_dict_file=None, name_base="circuit_", verbose=False):

      if list_dict_file is None:
         return {"error": "no 'list' or 'dict' or 'file' with jobids provided"}

      jobdict = {} # form: {"name": "UID"} with "name" different for each job
      jobidlist = []

      if isinstance(list_dict_file, dict):

        if verbose: print("'get_jobids_from_input' dict keys:",list_dict_file.keys())

        if "name" in list_dict_file.keys():
           name_base = list_dict_file["name"]

        if "ids" in list_dict_file.keys():
           jobidlist = list_dict_file["ids"]

        elif "id" in list_dict_file.keys():       # single "id"?
           if isinstance(list_dict_file["id"],list):
              jobidlist = list_dict_file["id"]
           else: 
              jobidlist = [list_dict_file["id"]]

        else:
           for k,v in list_dict_file.items():
              outdict = self.validate_jobid_hash(v)
              if "error" in outdict.keys():
                 if verbose: print("ERROR:", outdict)
              else:
                 jobdict.update({k: outdict["id"]})

           if len(jobdict) == 0:
              return {"error": f' dict "{list_dict_file}" does not contain keys "ids" or "id"'}

      elif isinstance(list_dict_file, list):

         jobidlist = list_dict_file

      elif isinstance(list_dict_file, str):

         if os.path.exists(list_dict_file):

            with open(list_dict_file, "r") as f: 
               lines = f.readlines()

               for j,id in enumerate(lines):

                  if id[0] != "#":
                     vals = id.strip().split(":")
                     outdict = self.validate_jobid_hash(vals[-1].strip())

                     if "id" in outdict.keys():
                        if len(vals) > 1:
                           jobdict.update({ vals[0].strip(): outdict["id"] })
                        else:
                           jobdict.update({ f"{name_base}{j}": outdict["id"] })

               if verbose: print("'get_jobids_from_input' got ids from file",jobdict)

         else:
            jobidlist = [list_dict_file]                        

      else:
         return {"error": f'"{list_dict_file}"" is not a list nor a dict nor a file with jobids'}

      if len(jobdict) == 0:

         for j,id in enumerate(jobidlist):
            outdict = self.validate_jobid_hash(id)
            if "error" in outdict.keys():
               if verbose: 
                  print(f'List entry {j} ("{id}") is not a valid jobid UID string')
            else:
               jobdict.update({ f"{name_base}{j}": outdict["id"] })

            if len(jobdict) == 0:
               return {"error": f"No valid jobid UID strings found in {list_dict_file}"}

      return jobdict

###############################################################################
# User functions:

   def retrieve_job(self, jobid=None, wait_minutes=0, verbose=False):
      """
     jobid can be a UID string or dict: {"id": jobid}
      """ 
      from qiskit_ionq.helpers import decompress_metadata_string_to_dict

      if jobid is None:
         jobid = self.last_jobid_dict
      if verbose: print(jobid, type(jobid))
      jobid = self.validate_jobid_hash(jobid)
      if "error" in jobid.keys():
         return jobid
      jobid = jobid["id"]

      flg_waitprint = True
      wait_minutes *= 2  #check in 30 sec intervals

      while wait_minutes >= 0:  

         response = requests.get(f'https://api.ionq.co/v0.3/jobs/{jobid}', headers=self.api_header)

         if response.ok:
            data = response.json()
            if verbose: print(f"GET returned: {data}")
            self.last_jobid_dict = data

            if data["status"] == "completed":
               results = requests.get(f"https://api.ionq.co{data['results_url']}", headers=self.api_header)

               if results.ok:

                  shots = float(data["shots"]) if "shots" in data.keys() else 1000
                  resdata = results.json()
                  if verbose: print("GET returned results:",resdata)

                  # returned 'counts' or 'probabilities'?
                  sumcnts = 0
                  for val in resdata.values():
                     sumcnts += val

                  flg_probs = sumcnts > 0.8 and sumcnts < 1.2 
                  probs={}; cnts={}
                  sumcnts = 0; sumprobs = 0.0 

                  for k,v in resdata.items():
                     kbin = bin(int(k))[2:].zfill(data["qubits"])
                     probs[kbin] = v if flg_probs else v/shots
                     cnts[kbin] = int(v*shots) if flg_probs else v
                     sumcnts  += cnts[kbin]
                     sumprobs += probs[kbin]

                  if sumprobs != 1.0:
                     for k,v in probs.items():
                        probs[k] = v/sumprobs

                  if sumcnts != shots:
                     maxk, maxcnt = '',0
                     for k,v in cnts.items():
                        if cnts[k] >= maxcnt:
                           maxk,maxcnt = k,cnts[k]
                     cnts[maxk] += int(shots - sumcnts)        

                  data["results"] = {"probabilities": probs, "counts": cnts}

                  if "metadata" in data.keys() and "qiskit_header" in data["metadata"].keys():
                     mydat = data["metadata"]["qiskit_header"]
                     if not isinstance(mydat, dict):
                        data["metadata"]["qiskit_header"] = decompress_metadata_string_to_dict(mydat)                                                      
                  self.last_jobid_dict = data
                  return data

               else:
                   print(f"Retrieved job information but no 'results': {results}")

            elif wait_minutes == 0:
               return data

         if flg_waitprint: 
            print(f"... up to {wait_minutes/2} min to go")
            flg_waitprint = False

         time.sleep(30)
         wait_minutes -= 1

      return {"error": f"{response}"}

###############

   def submit_job(self, circuit_or_file, wait_minutes=0, verbose=False, **kwargs):
      """
      circuit_or_file: qiskit or cirq circuit or json file/string or dictionary or string with QASM code.
      keys provided as argument overwrite entries in 'circuit_or_file'
      """

      argtype ='unknown'; datastr = ''

      if isinstance(circuit_or_file, str):

         if os.path.isfile(circuit_or_file):

            argtype = 'file'

            with open(circuit_or_file) as f:
#                cdat = f.read().replace('\n', '').replace('\r', '').replace('\t', '').replace(' ','').encode()
               datastr = re.sub('[\r\t\n ]','', f.read())

         elif circuit_or_file.find("OPENQASM 2.0", 0, 80) >= 0:

            argtype = 'qasm_string'
            datastr = self.translate_qasm(circuit_or_file, verbose)

            if datastr is None:
                return({"error": f"String does not contain an OPENQASM circuit: {circuit_or_file}"})

         elif circuit_or_file.find("{")>=0 and circuit_or_file.find(":")>0 and circuit_or_file.find("}")>0:

            argtype = 'string'
            datastr = circuit_or_file #re.sub(re.sub('[\r\t\n ]','', circuit_or_file))

      elif isinstance(circuit_or_file, list): 

         argtype = 'qasm_list'
         datastr = self.translate_qasm(circuit_or_file, verbose)

         if datastr is None:
            return({"error": f"List does not contain an OPENQASM circuit: {circuit_or_file}"})

      elif isinstance(circuit_or_file, dict):

         argtype = 'dict'
         datastr = json.dumps(circuit_or_file)

      else:

         # qiskit or cirq quantum circuit as input?
         if self.qiskit_installed > 0:
            import qiskit

            if isinstance(circuit_or_file, qiskit.QuantumCircuit):
            
               if self.qiskit_installed > 1:

                  from qiskit_ionq.helpers import qiskit_circ_to_ionq_circ, qiskit_to_ionq

                  if "backend" in kwargs.keys() and not isinstance(kwargs["backend"], str): 

                     backend = kwargs.pop("backend")
                     argtype = 'qiskitCircuit2ionq'
                     datastr = qiskit_to_ionq(circuit_or_file, backend)

                  else:                         

                     argtype = 'qiskitCircuit'
                     circ, qbits,_ = qiskit_circ_to_ionq_circ(circuit_or_file)
                     datastr = json.dumps({"body": { "gateset": "qis", "qubits": qbits, "circuit": circ } })         
     
               else:

                  #translate QASM circuit
                  argtype = 'qiskitQASMcircuit'
                  datastr = self.translate_qasm(circuit_or_file.qasm(), verbose)

         if len(datastr) < 1 and self.cirq_installed > 0:    

            import cirq

            if isinstance(circuit_or_file, cirq.Circuit):

               if self.cirq_installed > 1:

                  import cirq_ionq

                  argtype = 'cirqCircuit2ionq'
                  circ = cirq_ionq.Serializer().serialize(circuit_or_file)
                  datastr = json.dumps({"body": circ.body, "metadata": circ.metadata})

                  if "backend" in kwargs.keys() and isinstance(kwargs["backend"], cirq_ionq.Service):
                     kwargs["backend"] = kwargs["backend"]._client.default_target

               else:

                  argtype = 'cirqQASMcircuit'
                  datastr = self.translate_qasm(cirq.qasm(circuit_or_file), verbose)

         if len(datastr) < 1:

            return {"error": f"translation of '{circuit_or_file}' (type={type(circuit_or_file)}) not implemented"}

      if verbose: print(argtype+":",datastr)

      # JSON requires double quotes, not single quotes
      # JSON does not allow trailing comma in dicts & lists
      datastr = datastr.replace("'",'"').replace(".]","]").replace(".}","}")

      try:

         data = json.loads(datastr)

      except ValueError:

         if argtype=='file':
            text="This is not a valid JSON file: {}".format(circuit_or_file)
         elif argtype=='string':
            text="This is not a valid JSON format: {}".format(circuit_or_file)
         elif argtype=='dict':
            text="This is not a valid dictionary: {}".format(circuit_or_file)
         elif argtype=='circuit' or argtype=='circuit2ionq':
            text="Circuit could not be translated to JSON format"
         else:
            text="Unknown input format: {}".format(circuit_or_file)
         return {"error": text}

      # check whether basic keys exists and add missing keys

      body_key = "body" if "body" in data.keys() else "input"
      if body_key not in data.keys():
         return {"error": "dictionary key 'body' is missing in {}".format(data)}

      if not isinstance(data[body_key], dict):
         return {"error": "entry 'body' must be a dictionary in {}".format(data)}

      if "qubits" not in data[body_key].keys():
         return {"error": "dictionary key 'qubits' is missing in {}".format(data)}

      if "circuit" not in data[body_key].keys():
         return {"error": "dictionary key 'circuit' is missing in {}".format(data)}

      if not isinstance(data[body_key]["circuit"], list):
         return {"error": "entry 'circuit' must be a list in {}".format(data)}    

      if "gateset" not in data[body_key].keys():
         data[body_key]["gateset"] = "qis"    

      datentries = {"lang": "json", "target": "simulator", "shots": 1024, "name": "qiskit_circuit"}
 
      for k,v in datentries.items():

         if k in kwargs.keys():
            data[k] = kwargs.pop(k)

         elif k not in data.keys() or data[k] is None:
            data[k] = v 

      if "backend" in kwargs.keys() and "target" not in kwargs.keys():
         backend = kwargs.pop("backend")
         if not isinstance(backend, str): backend = backend.name()

         data["target"] = backend[5:] if backend.startswith("ionq_") else backend

      if "noise" in kwargs.keys():
         data["noise"] = kwargs.pop("noise") if isinstance(kwargs["noise"], dict) else {"model": kwargs.pop("noise")}

      elif "noise_model" in kwargs.keys():
         data["noise"] = {"model": kwargs.pop("noise_model")}

      if data["target"] == "simulator":

         if "noise" not in data.keys():
            data["noise"] = {"model": "ideal"}

         else:

            val = data["noise"]["model"]
            if val.startswith("qpu"):
                data["noise"]["model"] = val[5:]
            if val.endswith("aria"):
                data["noise"]["model"] = "aria-1"
            
         if "sampler_seed" in kwargs.keys():
            data["sampler_seed"] = kwargs["sampler_seed"]

         if "sampler_seed" in data.keys():
            data["noise"].update({"seed": data["sampler_seed"]})

      else:

         if "noise" in data.keys() and data["target"] == "qpu":
            v = data.pop("noise")
            data["target"] = f'qpu.{v["model"]}'

         if data["target"] == "harmony":
            data["target"] = "qpu.harmony"

         if data["target"] in ["aria", "aria-1", "qpu.aria"]:
            data["target"] == "qpu.aria-1"

      if "metadata" in data.keys(): 

         #skip this (not used by IonQ API)
         #if "qiskit_header" in data["metadata"].keys():
         #   data["metadata"].pop("qiskit_header")

         if "shots" in data["metadata"].keys():
            data["metadata"]["shots"] = str(data["shots"])

         if "sampler_seed" in data["metadata"].keys(): 

            if "sampler_seed" in data.keys():
               data["metadata"]["sampler_seed"] = str(data["sampler_seed"])
            else:
               data["metadata"]["sampler_seed"] = str(data["metadata"]["sampler_seed"])
    
      if not argtype.startswith('circuit') and not argtype.startswith('qasm'):

         checked_circuit = self.validate_circuit(data[body_key]["circuit"])
         if verbose: 
            print(f'validate_circuit returns {checked_circuit} ({type(checked_circuit)})')

         if "error" in checked_circuit.keys():
            checked_circuit.update({"data": datastr})
            return checked_circuit

         if "updated_entries" in checked_circuit.keys():
            mydat = checked_circuit.pop("updated_entries")
            data[body_key].update(checked_circuit)
            if verbose:
                print(f"validate_circuit: corrected gates in 'circuit' entries {mydat}")

      if verbose: print("before send:",data)

      header = self.api_header
      header.update({"Content-Type": "application/json"})

      response = requests.post('https://api.ionq.co/v0.3/jobs', headers=header, json=data, verify=False)

      if response.ok:

         outdata = response.json()
         if verbose: print(f"POST returned: {outdata}")

         #try to read results ...
         if wait_minutes > 0:

            results = self.retrieve_job(outdata, wait_minutes=wait_minutes, verbose=verbose)

            if 'error' in results.keys():
                outdata.update(results)
            else:
                outdata = results

         if outdata["status"] in ["submitted", "ready"] and data["target"] != "simulator":

            linetext = f"{datetime.now().strftime('%Y.%m.%d-%H:%M:%S')} : {outdata['id']}\n"

            if "save_file" in kwargs.keys():

               if isinstance(kwargs["save_file"], bool):

                  jobid_file_base = self.jobid_file_base if kwargs["save_file"] else ""

               elif isinstance(kwargs["save_file"], str):

                  jobid_file_base = kwargs["save_file"]

               elif isinstance(kwargs["save_file"], io.TextIOBase): 

                  f = kwargs["save_file"]
                  if f.closed:
                     open(f.name,"a").write(linetext)

                  else:
                     f.write(linetext)

                  jobid_file_base = ""

            else:

               jobid_file_base = self.jobid_file_base

            if len(jobid_file_base) > 0:

               if len(jobid_file_base) > 4 and jobid_file_base.find('.', -5) >= 0:

                  ioflag = "a" if os.path.exists(jobid_file_base) else "w"

                  with open(jobid_file_base, ioflag) as f:
                     f.write(linetext)

               else:

                  if not os.path.exists(f"{jobid_file_base}.txt"):

                     with open(f"{jobid_file_base}.txt", "w") as f:
                        f.write(linetext)

                  else:

                     j = 0
                     while os.path.exists(f"{jobid_file_base}{j}.txt"):
                        j +=1

                     with open(f"{jobid_file_base}{j}.txt", "w") as f:
                        f.write(linetext)            

         self.last_jobid_dict = outdata
         return outdata

      else:

         errtxt = f"{response}"
         if response.status_code == 400:
            errtxt += f": Bad request - bad format in {circuit_or_file}"

         elif response.status_code == 401:
            errtxt += ": Authentication failed"

         if verbose: print("ERROR:", errtxt)
         return {"error": errtxt}

###############

   def cancel_job(self, jobid=None, verbose=False):

      if jobid is None:
         jobid = self.last_jobid_dict

      jobid_dict = self.validate_jobid_hash(jobid)

      if verbose: print("cancel_job: validate_jobid_hash returned:",jobid_dict)

      if "error" in jobid_dict.keys():

         if isinstance(jobid, list):

            errtxt = "Use function: 'cancel_multiple_jobs' instead"

            if len(jobid) == 1:
               jobid_dict = self.validate_jobid_hash(jobid[0])

            else:
               jobid_dict.update({"NOTE": errtxt})

         elif isinstance(jobid, str) and os.path.exists(jobid):
            jobid_dict.update({"NOTE": errtxt})

         if "error" in jobid_dict.keys():
            return jobid_dict
        
      if verbose: print("before GET",jobid_dict)

      response = requests.get(f'https://api.ionq.co/v0.3/jobs/{jobid_dict["id"]}', headers=self.api_header)

      if response.ok:

         data = response.json()

         if "status" not in data.keys():
            return {"error": f"No 'status' information retrieved for job {jobid_dict['id']}"}
 
         if data["status"] in ['failed','running','completed','canceled']:
            data.update({"NOTE": "Only jobs that are 'submitted' or 'ready' are being cancelled"})
            return data

         else:
            response = requests.put(f'https://api.ionq.co/v0.3/jobs/{jobid_dict["id"]}/status/cancel', headers=self.api_header)

            if response.ok:
               return response.json()

            else:
               errtxt = f"Request to cancel job {jobid_dict['id']} returned {response}"
               return {"error": errtxt}

      else:

         errtxt = f"{response}"
         if response.status_code == 401:
            errtxt += ": Authentication failed"
         return {"error": errtxt}

#############################################################
# User function for multiple job submission, retrieval, cancelling
    
   def retrieve_multiple_jobs(self, list_dict_file=None, wait_minutes=0, verbose=False):
      #input: list of jobids (list or dict or file)

      if list_dict_file is None:

         print("Usage:\nretrieve_multiple_jobs(list_of_circuits, wait_minutes[=0], verbose[=False])")
         print("     list_of_circuits: list of QuantumCircuits or JSON-formatted circuits (dicts or strings or files)")
         print("              or dict of form {'name_of_circuit': JSON-formatted circuit or QuantumCircuit}")
         return {"error": "no input list or dict of circuits provided"}
 
      jobdict = self.get_jobids_from_input(list_dict_file, verbose=verbose)

      if "error" in jobdict.keys():
         if verbose: print("ERROR in 'retrieve_multiple_jobs':", jobdict["error"])
         return jobdict

      if verbose: print("get_jobids_from_input returned:", jobdict)

      outdict = {}
      for name, jobid in jobdict.items():

         if verbose: print(name, jobid)
         data = self.retrieve_job(jobid, wait_minutes=wait_minutes, verbose=verbose)
         outdict.update({f"{name}": data})

      return outdict

########################

   def submit_multiple_jobs(self, list_of_circuits=None, wait_minutes=0, verbose=False, **kwargs):
      #input: list of JSON-formatted circuits (dicts or strings or files) or list of QuantumCircuits
      #or dict of form {"name_of_circuit": JSON-formatted circuit or QuantumCircuit}

      if list_of_circuits is None:

         print("Usage:\nsubmit_multiple_jobs(list_of_circuits, wait_minutes[=0], verbose[=False], ...)")
         print("     list_of_circuits: list of QuantumCircuits or JSON-formatted circuits (dicts or strings or files)")
         print("              or dict of form {'name_of_circuit': JSON-formatted circuit or QuantumCircuit}")
         print("     ...: additional circuit execution parameters, e.g. shots=NNN, target='simulator', etc. ")
         return {"error": "no input list or dict of circuits provided"}

      name = kwargs.pop("name") if "name" in kwargs.keys() else "circuit"  

      # not a list but single job?
      if not isinstance(list_of_circuits, list):
         jobdict = self.submit_job(list_of_circuits, wait_minutes=wait_minutes, verbose=verbose, name=name, **kwargs)

         if "error" in jobdict.keys():
            return jobdict
 
         return {f"{name}": jobdict}

      names = name if isinstance(name, list) else [f"{name}_{j}" for j in range(len(list_of_circuits))]

      if "name_base" in kwargs.keys():
         names = [f"{kwargs['name_base']}{j}" for j in range(len(list_of_circuits))]
         kwargs.pop("name_base")

      elif "names" in kwargs.keys() and isinstance(kwargs["names"], list):
         names = kwargs.pop("names")

      if len(set(names)) != len(list_of_circuits):

         if isinstance(name, list): name = name[0]
         if verbose: print(f"Change names to {name}_N because list of names has different length "
                          f"({len(names)}) than list of circuits ({len(list_of_circuits)})")
         names = [f"{name}_{j}" for j in range(len(list_of_circuits))]

      outdict = {}; filebase = ""

      if "save_file" in kwargs.keys():

         if isinstance(kwargs["save_file"], bool):

            filebase = self.jobid_file_base if kwargs["save_file"] else ""

         elif isinstance(kwargs["save_file"], io.TextIOBase): 

            filebase = ""
         
         else:

            filebase = str(kwargs["save_file"])

      else:         

         filebase = self.jobid_file_base

      if len(filebase) > 0:

         ioflag = "w"

         if len(filebase) > 4 and filebase.find(".", -5) >= 0:

            if os.path.exists(filebase):
               ioflag = "a"

         else:

            if os.path.exists(f"{filebase}.txt"):

               j = 0
               while os.path.exists(f"{filebase}{j}.txt"):
                  j +=1 

               filebase = f"{filebase}{j}.txt"

            else:

               filebase += ".txt"

         kwargs["save_file"] = open(filebase, ioflag)

      for nam,circ in zip(names,list_of_circuits):

         job = self.submit_job(circ, wait_minutes=0, verbose=verbose, name=nam, **kwargs)
         if "error" in job.keys():
            if verbose: print(f"ERROR:",job["error"])

         outdict.update({nam: job})

      if len(filebase) > 1:

         if isinstance(kwargs["save_file"], io.TextIOBase):

            f = kwargs["save_file"]
            if not f.closed:
               f.close()

            with open(f.name) as fp:
               nchars = len(fp.readlines())

            if nchars < 20:
               os.remove(fp.name)

      if wait_minutes > 0:
         outdict = self.retrieve_multiple_jobs(outdict, wait_minutes=wait_minutes, verbose=verbose)

      return outdict

########################

   def cancel_multiple_jobs(self, list_dict_file=None, verbose=False):
      #input be a list or dict of jobids or file containing one jobid per line

      if list_dict_file is None:
         print("Usage:\ncancel_multiple_jobs(list_dict_file, verbose[=False])")
         print("     list_or_file: list or dict of jobids or file containing one jobid per line")
         return {"error": "no input list or file with jobids provided"}

      jobdict = self.get_jobids_from_input(list_dict_file, verbose=verbose)
      if verbose: print("after get_jobids_from_input:",jobdict)

      jobids = [id for k,id in jobdict.items() if k != "error"]
      if len(jobids) == 0:
         return jobdict

      jsonstr = '{ "ids": ["'
      jsonstr += '","'.join(jobids)
      jsonstr += '"] }'
      if verbose: print(jsonstr)
      
      header = self.api_header
      header.update({"Content-Type": "application/json"})

      response = requests.put('https://api.ionq.co/v0.3/jobs/status/cancel', headers=header, json=jsonstr, verify=False)

      if verbose: print(f"PUT returned: {response}")

      if response.ok:
         data = response.json()
         if verbose: print("PUT data:", data)
         ncancelled = len(data["ids"])

      else:
         ncancelled = 0
         datalist = []
         for id in jobids:
            outdict = self.cancel_job(id, verbose)
            if verbose: print(outdict)
            if "status" in outdict.keys() and outdict["status"] == "canceled":
               ncancelled +=1
               datalist.append(outdict["id"])
 
         data = {"ids": datalist}

      print(f"Cancelled {ncancelled} jobs from {jobdict}")

      return data

##########################
# User function to retrieve results from returned dictionary

   def extract_probabilities(self, jobdict=None):
      """
   Extract probabilities for all results (single or multiple retrieved jobs)
      """

      if jobdict is None:
         jobdict = self.last_jobid_dict

      if isinstance(jobdict, list) or isinstance(jobdict, str):

         if isinstance(jobdict, str):
            jobdict = self.retrieve_job(jobdict)

         else:
            jobdict = self.retrieve_multiple_jobs(jobdict)

         if "error" in jobdict.keys():
            print("ERROR: input is not a dictionary (use the returned dict of 'retrieve_job' or 'retrieve_multiple_jobs' as input) ")
            return None

      if not isinstance(jobdict, dict):
         print("ERROR: input is not a dictionary (use the returned dict of 'retrieve_job' or 'retrieve_multiple_jobs' as input) ")
         return None

      if "id" in jobdict.keys() and "status" in jobdict.keys():  #assume single job

         if jobdict["status"] in ["submitted", "ready"]:
            jobdict = self.retrieve_job(jobdict)

         if jobdict["status"] == "completed":
            return jobdict["results"]["probabilities"]
 
         else:
            print("Job '{}' has status '{}'".format(jobdict["id"],jobdict["status"]))
            return None

      else:  #multiple jobs

         outdict = {}
         for k,v in jobdict.items():

            if isinstance(v, dict):

                if v["status"] in ["submitted", "ready"]:
                   v = self.retrieve_job(v)

                if "error" in v.keys() or v["status"] != "completed":
                   print(f"{k}:", v)
                else: 
                   outdict.update({f"{k}:": v["results"]["probabilities"]})

            else:
               print(f"{k}:", v)

         return outdict if len(outdict)>0 else None

###################
    
   def extract_counts(self, jobdict=None):
      """
   Extract counts for all results (single or multiple retrieved jobs)
      """

      if jobdict is None:
         jobdict = self.last_jobid_dict

      if isinstance(jobdict, list) or isinstance(jobdict, str):

         if isinstance(jobdict, str):
            jobdict = self.retrieve_job(jobdict)

         else:
            jobdict = self.retrieve_multiple_jobs(jobdict)

         if "error" in jobdict.keys():
            print("ERROR: input is not a dictionary (use the returned dict of 'retrieve_job' or 'retrieve_multiple_jobs' as input) ")
            return None

      if not isinstance(jobdict, dict):
         print("ERROR: input is not a dictionary (use the returned dict of 'retrieve_job' or 'retrieve_multiple_jobs' as input) ")
         return None

      if "id" in jobdict.keys() and "status" in jobdict.keys():  #assume single job

         if jobdict["status"] in ["submitted", "ready"]:
            jobdict = self.retrieve_job(jobdict)

         if jobdict["status"] == "completed":

            if "results" in jobdict.keys():
               if "counts" in jobdict["results"].keys():
                  return jobdict["results"]["counts"]

               elif "shots" in jobdict.keys(): 

                  shots = jobdict["shots"]

                  outdict = {}
                  if "probabilities" in jobdict["results"].keys():
                     mydict = jobdict["results"]["probabilities"]

                  else:
                     mydict = jobdict["results"]

                  for k,v in mydict:
                     outdict.update({k:int(v*shots)})
                  return outdict

            print("Job '{}' does not have result counts nor probabilities listed")
            return None
                
         else:
 
            print("Job '{}' has status '{}'".format(jobdict["id"],jobdict["status"]))
            return None

      else:  #multiple jobs

         outdict = {}
         for k,v in jobdict.items():

            if isinstance(v, dict):
               if v["status"] in ["submitted", "ready"]:
                  v = self.retrieve_job(v)

               if "error" in v.keys() or v["status"] != "completed":
                  print(f"{k}:", v)

               else: 
                  outdict.update({f"{k}:": v["results"]["counts"]})

            else:
                print(f"{k}:", v)

         return outdict if len(outdict)>0 else None

#--------------------

   def get_waittime(self, backend="ACCESS", calib_data=False):
      """
get_waittime(backend, data) 
   - print current wait time (average queue time) for provided backend
   input arguments:
    backend: for which IonQ API backend to print: "simulator" or "qpu.harmony" or "qpu.aria-1" ...
       backend="ALL": print wait time, status, and access information for all existing backends
       backend="ACCESS" (default): print wait time and status for all backends, to which we have access
    calib_data: flag on whether also to print calibration/characterization results
      """
      special_vals = ["ALL", "ACCESS"]

      if isinstance(backend, str):
         backend = [backend]

      for j,val in enumerate(backend):
         if val not in special_vals:
            val = val.lower()
            if val in ["qpu.aria", "aria"]:
               val = "qpu.aria-1"
            if val == "harmony":
               val = "qpu.harmony"
            backend[j] = val
                
      response = requests.get("https://api.ionq.co/v0.3/backends", headers = self.api_header)

      if not response.ok:

         errtxt = f"{response}"
         if response.status_code == 400:
            errtxt += f": Bad request"

         elif response.status_code == 401:
            errtxt += ": Authentication failed"

         print("ERROR accessing IonQ Server using provided IonQ API token:", errtxt)
         return

      data = response.json()
      
      calibinfo = {}
      
      print(" Name              Qubits   Status   Access?  Degraded?  avg. Queue Time  (last updated)")

      for mydict in data:
         name = mydict["backend"]

         if backend[0] in special_vals or name in backend:
         
            if backend[0] == "ACCESS" and int(mydict["has_access"]) == 0: 
               continue

            waitsecs = mydict["average_queue_time"]//100 if name=="simulator" else mydict["average_queue_time"]//1000 
            waittime = str(timedelta(seconds=waitsecs))      
            timestamp = datetime.fromtimestamp(mydict["last_updated"])
            has_access = "Yes" if mydict["has_access"] else "No"
            degraded = "Yes" if mydict["degraded"] else "No"
            blanks = " " * (16 - len(name))

            print(" {}: {} {} {:>12s}  {:>5s}   {:>5s}   {:>19s}  ({})".format(name, blanks, mydict["qubits"], mydict["status"], has_access, degraded, waittime, timestamp))

            if calib_data:
               if name == "simulator":
                  calibinfo.update({ name: mydict["noise_models"] })
               elif "characterization_url" in mydict:
                  calibinfo.update({ name: mydict["characterization_url"] })

      if calib_data and len(calibinfo) > 0:

         print("\nCalibration data:")

         for k,v in calibinfo.items():

            blanks = " " * (14 - len(k))

            if k == "simulator":
               print("{}: {} noise_models: {}".format(k, blanks, v))

            else:

               response = requests.get(f"https://api.ionq.co/v0.3{v}", headers = self.api_header)

               if response.ok:
                  data = response.json()
                  fidinfo = data["fidelity"]
                  timinfo = data["timing"]

                  print("{}: {} fidelity: 1q-gates {:6.4f}, 2q-gates {:6.4f}, prep+meas. {:6.4f}".format(k, blanks, fidinfo["1q"]["mean"], fidinfo["2q"]["mean"], fidinfo["spam"]["mean"]))
                  print("        timing (in msec): 1q-gates {:6.4f}, 2q-gates {:6.4f}, readout {:6.4f}, reset {:6.4f}".format(timinfo["1q"]*1000, timinfo["2q"]*1000, timinfo["readout"]*1000, timinfo["reset"]*1000))
                  print("        (date: {})".format(datetime.fromtimestamp(data["date"])))
