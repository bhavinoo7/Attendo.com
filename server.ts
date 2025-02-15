import http from "http";
import express from "express";
import next from "next";
import { Server as socketIo } from "socket.io";
import Student from "@/model/Student";
import { StudentSession } from "@/model/Student";
import { Teacher, session } from "./src/model/Teacher"; // Corrected import path
import UserModel from "./src/model/User";
import mongoose, { model, ObjectId } from "mongoose"; // Corrected import path
import path from "path";
import { Batch, Division } from "@/model/Division";
import { Attendance } from "@/model/Timetable";

const cluster = require("cluster");
const os = require("os");
const numCPUs = os.cpus().length;

// Initialize Next.js
const connectToDatabase = async () => {
  try {
    await mongoose.connect(
      "mongodb+srv://bhavin:bhavin@cluster0.cs8ai.mongodb.net/"
    );
    console.log("Connected to MongoDB");
  } catch (error) {
    console.error("Error connecting to MongoDB:", error);
  }
};
connectToDatabase();

const dev = process.env.NODE_ENV !== "production";
const app = next({ dev });
const handle = app.getRequestHandler();
const server = express();

app.prepare().then(() => {

  const httpServer = http.createServer(server);

  // Initialize Socket.IO with the HTTP serve
  const io = new socketIo(httpServer, {
    path: "/api/socket",
    cors: {
      origin: "*", // Adjust according to your needs
      methods: ["GET", "POST"],
    },
  });

  // Handle WebSocket events
  io.on("connection", (socket) => {
    console.log("New client connected:", socket.id);

    socket.on("post-teacher-data", async (data) => {
      console.log("Received data:", data);
      if (data !== null && data.length > 0) {
        const user1 = await UserModel.findById(data);
        const user = await UserModel.aggregate([
          {
            $match: {
              _id: new mongoose.Types.ObjectId(data),
            },
          },
          {
            $lookup: {
              from: "teachers",
              localField: "teacherid",
              foreignField: "_id",
              as: "teacherResult",
            },
          },
          {
            $lookup: {
              from: "slots",
              localField: "teacherResult.lectures",
              foreignField: "_id",
              as: "slotResult",
            },
          },
          {
            $unwind: "$slotResult", // Unwind the slotresult array
          },
          {
            $lookup: {
              from: "divisions",
              localField: "slotResult.division_id",
              foreignField: "_id",
              as: "division",
            },
          },
          {
            $unwind: "$division", // Unwind the slotresult array
          },
          {
            $addFields: {
              "slotResult.division_name": "$division.division_name",
            },
          },
          {
            $group: {
              _id: "$_id",
              slots: { $push: "$slotResult" },
            },
          },
          {
            $project: {
              slots: 1,
            },
          },
        ]);
        
        if (user[0].slots) {
          user[0].slots.map(async (slot: any) => {
           
            if (slot.is_lab) {
              console.log("lab");
              const result = slot.lab.filter((lab: any) => {
                return lab.teacher.toString() === user1?.teacherid.toString();
              });
             
              if (result.length > 0) {
                slot.labs = result;
                
              } else {
                slot.labs = [];
              }

              
              const batch = await Batch.findById(slot.labs[0].batch_id);
             
              if (batch) {
                slot.students = batch.students;
              } else {
                slot.students = [];
              }
            } else {
              console.log("lecture");
              const division = await Division.findById(slot.division_id);
            
              if (division) {
                slot.students = division.students;
              } else {
                slot.students = [];
              }
            }
          });
        }
        
        socket.emit("get-teacher-data", { data: user[0].slots });
      }
    });

    socket.on("get-session", async (data) => {
      const { id } = data;
      console.log("AAAAA", data);
      if (data.id) {
        console.log("BBBBB", id);
        const user = await UserModel.findById(id);
        
        const teacher = await Teacher.findById(user?.teacherid);
       
       
        if (teacher.currentent_session !== null) {
          socket.emit("get-session", { data: teacher.currentent_session });
        } else {
          socket.emit("session-not-found");
        }
      }
    });
    socket.on("get-student-session", async (data) => {
      const { id,studentid } = data;
      console.log("AAAAA", data);
      const sessio = await session.find({ _id: id, status: "open" }).populate({path:"Attendance",model:Attendance});
      console.log(sessio);
      if (sessio.length > 0) {
        console.log("inner");
        socket.emit("get-student-session", sessio);
      } else {
        socket.emit("session-not-found");
      }
    });
    socket.on("get-student-details", async (data) => {
      const { id } = data;
      console.log("AAAAA", data);
      if (id && id.length > 0) {
        const user = await UserModel.findById(id);
        const student = await Student.findById(user?.studentid);
        
        socket.emit("get-student-details", student);
      }
    });

    socket.on("make-attendance", async (data) => {
      console.log("BAAAA", data);
      const s1 = await session.findById(data.session_id).populate({path:"Attendance",model:Attendance});
     
      const teacher = await Teacher.find({ email: data.teacher_email });
      
      const student = await Student.findById(data.student_id);
      

      if (teacher.length > 0 && s1 !== null) {
        console.log("now in");
        if (
          s1.status === "open" &&
          teacher[0].currentent_session.toString() === data.session_id
        ) {
          let distance = checkStudentDistance(
            data.student_location,
            s1.location
          );
          console.log(distance);
          if (s1.Attendance.length > 0) {
            s1.Attendance.map(async (attendance: any) => {
              if (attendance.student_id.toString() === data.student_id) {
                if (attendance.status === "Present") {
                  socket.emit(
                    "attendance-response",
                    "Attendance Already marked"
                  );
                } else {
                  const ssession = new StudentSession({
                    session_id: data.session_id,
                    student_id: data.student_id,
                    IP: data.IP,
                    image: data.image,
                    student_location: data.student_location,
                    distance: distance,
                    status: "Present",
                    date: new Date(),
                    time: new Date().toLocaleTimeString(),
                  });
                  ssession.save();
                  
                  student?.sessions.push(ssession._id as ObjectId);
                  await student?.save();
                  console.log("student",student);
                  attendance.status = "Present";
                  attendance.IP = data.IP;
                  attendance.image = data.image;
                  attendance.location = data.student_location;
                  attendance.distance = distance;
                  attendance.date = new Date();
                  attendance.save();
                  io.emit(
                    "attendance-response",
                    "Attendance marked successfully"
                  );
                  socket.emit("Response-done");
                }
              } else {
                console.log("not found");
              }
            });
          }
        }
      }
    });

    socket.on("fetch-student-sessions", async (data) => {
      console.log("AAAAA", data);
      
      if(data && data.length>0){
      const studentdata=await Student.aggregate([
        {
          $match: {
            _id: new mongoose.Types.ObjectId(data),
          },
        },
        {
          $lookup: {
            from: "studentsessions",
            localField: "sessions",
            foreignField: "_id",
            as: "sessionResult",
          },
        },
        {
          $unwind: "$sessionResult", // Unwind the slotresult array
        },
        {
          $lookup: {
            from: "sessions",
            localField: "sessionResult.session_id",
            foreignField: "_id",
            as: "session",
          },
        },
        {
          $unwind: "$session", // Unwind the slotresult array
        },
        {
          $addFields: {
            "sessionResult.session_name": "$session.session_name",
          },
        },
        {
          $group: {
            _id: "$_id",
            sessions: { $push: "$sessionResult" },
          },
        },
        {
          $project: {
            sessions: 1,
          },
        },

      ])
      
      socket.emit("fetch-student-sessions",studentdata[0].sessions)
    }
    });

    socket.on("get-live-attendance", async (data) => {
      console.log("get-live-attendance", data);
      
      if(data && data.length>0){
        console.log("teacherid",data); 
        const teacher =await Teacher.aggregate([
          {
            $match: {
              _id: new mongoose.Types.ObjectId(data),
            },
          },
          {
            $lookup: {
              from: "sessions",
              localField: "currentent_session",
              foreignField: "_id",
              as: "result",
            },
          },
          {
            $unwind: "$result", // Unwind the slotresult array
          },
          {
            $lookup: {
              from: "attendances",
              localField: "result.Attendance",
              foreignField: "_id",
              as: "result2",
            },
          },
          {
            $unwind: "$result2", // Unwind the slotresult array
          },
          {
            $lookup: {
              from: "students",
              localField: "result2.student_id",
              foreignField: "_id",
              as: "student",
            },
          },
          {
            $unwind: "$student", // Unwind the slotresult array
          },
          {
            $addFields: {
              "result2.student_name": "$student.name",
              "result2.student_enroll": "$student.enroll_no",
            },
          },
          {
            $group: {
              _id: "$_id",
              Attendance: { $push: "$result2" },
            },
          },
          {
            $project: {
              Attendance: 1,
            },
          },
        ])
       
        socket.emit("post-live-attendance",teacher[0].Attendance);
      }
    });

    socket.on("attendance-status",async(data)=>{
      const {session_id,student_id}=data;
      if(data.session_id && data.student_id)
      {
        const sessions=await session.findById(session_id).populate({path:"Attendance",model:Attendance});
        console.log(sessions);
        let s1=sessions.Attendance.filter((session:any)=>{
          return session.student_id==student_id;
        })
        console.log("mb",s1);
        if(s1[0].status=="Present")
        {
          socket.emit("attendance-status");
        }
      }
    })


    socket.on("disconnect", () => {
      console.log("Client disconnected:", socket.id);
    });
  });

  // Handle all Next.js requests
  server.all("*", (req, res) => {
    return handle(req, res); // Let Next.js handle the routing
  });

  // Start the server on the desired port
  const PORT = process.env.PORT  ;
  httpServer.listen(PORT, () => {
    console.log(`> Ready on http://localhost:${PORT}`);
  });

});


// Add this line to ensure the file is treated as an ES Module
export {};

function haversineDistance(lat1: any, lon1: any, lat2: any, lon2: any) {
  const R = 6371000; // Radius of the Earth in meters
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  const distance = R * c; // Distance in meters
  return distance;
}
function checkStudentDistance(Location1: any, Location2: any) {
  Location1 = Location1.split(",");
  Location2 = Location2.split(",");
  const locationLat1 = parseFloat(Location1[0]);
  const locationLon1 = parseFloat(Location1[1]);
  const locationLat2 = parseFloat(Location2[0]);
  const locationLon2 = parseFloat(Location2[1]);

  const distance = haversineDistance(
    locationLat1,
    locationLon1,
    locationLat2,
    locationLon2
  );
  return distance.toFixed(2);
}


