# Kernel
A [[Kernel]] is a component of an operating system that manages and allocates computer resources.
![Kernal|300](https://media.geeksforgeeks.org/wp-content/uploads/20250124124411692602/kernel.webp)

It is possible to run programs on OS without a kernal. However, [[Kernel]] greatly simplifies the writing and use of other programs.

---
## Task performed by Kernal

![Kernal Tasks|300](https://media.geeksforgeeks.org/wp-content/uploads/20250725113119973417/Function-Of-Kernel.webp)

- **Process Scheduling**: Linux is a preemptive multitasking operating system.
	- **Multitasking**: Multiple processes can simultaneously reside in memory. Each may receive use of the CPU(s).
	- **Preemptive**: Rules governing which process receives use of CPU, and for how long are determined by the [[Kernel|kernal process scheduler]].
- **Memory Management**: [[Kernel]] share RAM memory among processes in equitable and efficient fashion using virtual memory.
	- Processes are isolated from one another and from the [[kernel]].
	- Only part of the process needs to be kept in memory.
	  This lowers the memory requirement of each process, and allow more processes to be held in RAM simultaneously.
- **Provision of file system**: The [[kernel]] provides a file system on disk, allowing files to be created, retrieved, updated, deleted.
- **Creation and Termination of process**: 
	- The [[kernel]] can load a new program into memory, providing it with the resources. 
	- Once a process has completed execution, the [[kernel]] ensures that the resources it uses are freed for subsequent reuse by later programs.
- **Access to Devices**: The [[kernel]] provides programs with
	- an interface that standardizes and simplifies access to devices
	- arbitrate access by multiple processes to each device.
- **Networking**: The kernel transmits and receives network messages (packets) on behalf of user processes. This task includes routing of network packets to the target system.
- **Provision of a system call API**

---
### Kernal Mode vs User Mode
Areas of virtual memory can be marked as being part of **user space**, or **kernal space**.
- In user mode, CPU can only access memory marked as user space. Attempt to access [[Kernel|kernel space]] results in hardware exception.
- In kernel mode, the CPU can access both kernel and user space.

---

