#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "drake/geometry/scene_graph.h"
#include "drake/geometry/collision_filter_declaration.h"
#include "drake/geometry/geometry_set.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/math/spatial_algebra.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/externally_applied_spatial_force.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/joint_actuator.h"
#include "drake/multibody/tree/model_instance.h"
#include "drake/multibody/tree/rigid_body.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/context.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"

namespace py = pybind11;

namespace {

constexpr int kRootQposDim = 7;
constexpr int kRootQvelDim = 6;

using drake::geometry::CollisionFilterDeclaration;
using drake::geometry::GeometrySet;
using drake::geometry::SceneGraph;
using drake::math::RigidTransform;
using drake::multibody::BodyIndex;
using drake::multibody::ContactModel;
using drake::multibody::DiscreteContactApproximation;
using drake::multibody::ExternallyAppliedSpatialForce;
using drake::multibody::JointActuatorIndex;
using drake::multibody::ModelInstanceIndex;
using drake::multibody::MultibodyPlant;
using drake::multibody::Parser;
using drake::multibody::PdControllerGains;
using drake::multibody::RigidBody;
using drake::multibody::SpatialForce;
using drake::systems::Context;
using drake::systems::Diagram;
using drake::systems::DiagramBuilder;
using drake::systems::Simulator;

py::array_t<double> MakeArray(const std::vector<py::ssize_t>& shape) {
  return py::array_t<double>(shape);
}

Eigen::VectorXd MujocoQposToDrake(const double* qpos, int nq) {
  Eigen::VectorXd out(nq);
  out.segment(0, 4) = Eigen::Map<const Eigen::Vector4d>(qpos + 3);
  out.segment(4, 3) = Eigen::Map<const Eigen::Vector3d>(qpos);
  if (nq > kRootQposDim) {
    out.segment(kRootQposDim, nq - kRootQposDim) =
        Eigen::Map<const Eigen::VectorXd>(qpos + kRootQposDim, nq - kRootQposDim);
  }
  return out;
}

Eigen::VectorXd MujocoQvelToDrake(const double* qvel, int nv) {
  Eigen::VectorXd out(nv);
  out.segment(0, 3) = Eigen::Map<const Eigen::Vector3d>(qvel + 3);
  out.segment(3, 3) = Eigen::Map<const Eigen::Vector3d>(qvel);
  if (nv > kRootQvelDim) {
    out.segment(kRootQvelDim, nv - kRootQvelDim) =
        Eigen::Map<const Eigen::VectorXd>(qvel + kRootQvelDim, nv - kRootQvelDim);
  }
  return out;
}

void DrakeQposToMujoco(const Eigen::VectorXd& qpos, double* out) {
  Eigen::Map<Eigen::Vector3d> pos(out);
  Eigen::Map<Eigen::Vector4d> quat(out + 3);
  pos = qpos.segment(4, 3);
  quat = qpos.segment(0, 4);
  if (qpos.size() > kRootQposDim) {
    Eigen::Map<Eigen::VectorXd> joints(out + kRootQposDim, qpos.size() - kRootQposDim);
    joints = qpos.segment(kRootQposDim, qpos.size() - kRootQposDim);
  }
}

void DrakeQvelToMujoco(const Eigen::VectorXd& qvel, double* out) {
  Eigen::Map<Eigen::Vector3d> linear(out);
  Eigen::Map<Eigen::Vector3d> angular(out + 3);
  linear = qvel.segment(3, 3);
  angular = qvel.segment(0, 3);
  if (qvel.size() > kRootQvelDim) {
    Eigen::Map<Eigen::VectorXd> joints(out + kRootQvelDim, qvel.size() - kRootQvelDim);
    joints = qvel.segment(kRootQvelDim, qvel.size() - kRootQvelDim);
  }
}

void RequireShape(const py::buffer_info& info, const std::vector<py::ssize_t>& shape,
                  const std::string& name) {
  if (info.ndim != static_cast<py::ssize_t>(shape.size())) {
    throw std::invalid_argument(name + " has wrong rank");
  }
  for (int i = 0; i < info.ndim; ++i) {
    if (info.shape[i] != shape[i]) {
      throw std::invalid_argument(name + " has wrong shape");
    }
  }
}

struct EnvRuntime {
  Context<double>* plant_context{};
  std::unique_ptr<Simulator<double>> simulator;
};

class NativeDrakeEnvPool {
 public:
  NativeDrakeEnvPool(const std::string& model_file, int nbatch, double sim_dt,
                     py::array_t<double, py::array::c_style | py::array::forcecast> ctrl_limits,
                     py::array_t<double, py::array::c_style | py::array::forcecast> torque_limits,
                     int base_body_index, int push_body_index,
                     const std::vector<int>& foot_body_indices,
                     py::array_t<double, py::array::c_style | py::array::forcecast> foot_offsets,
                     double kp, double kd, int nthread)
      : nbatch_(nbatch),
        sim_dt_(sim_dt),
        ctrl_limits_(std::move(ctrl_limits)),
        torque_limits_(std::move(torque_limits)),
        foot_body_indices_(foot_body_indices),
        foot_offsets_(std::move(foot_offsets)),
        kp_(kp),
        kd_(kd),
        nthread_(std::max(1, nthread)) {
    if (nbatch_ < 1) {
      throw std::invalid_argument("nbatch must be >= 1");
    }
    auto ctrl_info = ctrl_limits_.request();
    if (ctrl_info.ndim != 2 || ctrl_info.shape[1] != 2) {
      throw std::invalid_argument("ctrl_limits must have shape (nu, 2)");
    }
    nu_ = static_cast<int>(ctrl_info.shape[0]);
    RequireShape(torque_limits_.request(), {nu_}, "torque_limits");
    RequireShape(foot_offsets_.request(), {static_cast<py::ssize_t>(foot_body_indices_.size()), 3},
                 "foot_offsets");

    DiagramBuilder<double> builder;
    auto [plant_ref, scene_graph_ref] =
        drake::multibody::AddMultibodyPlantSceneGraph(&builder, sim_dt_);
    plant_ = &plant_ref;
    scene_graph_ = &scene_graph_ref;
    plant_->set_contact_model(ContactModel::kPointContactOnly);
    plant_->set_discrete_contact_approximation(DiscreteContactApproximation::kSap);
    const auto model_instances = Parser(plant_).AddModels(model_file);
    if (model_instances.size() != 1) {
      throw std::runtime_error("NativeDrakeEnvPool expected exactly one model instance");
    }
    model_instance_ = model_instances.at(0);

    auto torque = torque_limits_.unchecked<1>();
    for (int i = 0; i < nu_; ++i) {
      auto& actuator = plant_->get_mutable_joint_actuator(JointActuatorIndex(i));
      actuator.set_effort_limit(torque(i));
      actuator.set_controller_gains(PdControllerGains(kp_, kd_));
    }

    // Match the Go1 pydrake backend's MJCF collision topology: Drake's parser
    // does not honor MuJoCo contype/conaffinity here, so exclude robot
    // self-collisions before Finalize().
    num_filtered_geometries_ = ExcludeRobotSelfCollisions();
    plant_->Finalize();
    trunk_body_ = &plant_->get_body(BodyIndex(base_body_index));
    push_body_ = &plant_->get_body(BodyIndex(push_body_index));
    for (int foot_body_index : foot_body_indices_) {
      foot_bodies_.push_back(&plant_->get_body(BodyIndex(foot_body_index)));
    }
    diagram_ = builder.Build();

    nq_ = plant_->num_positions();
    nv_ = plant_->num_velocities();
    state_dim_ = 1 + nq_ + nv_;
    if (nu_ != plant_->num_actuators()) {
      throw std::runtime_error("ctrl_limits length does not match plant actuators");
    }
    envs_.reserve(nbatch_);
    for (int i = 0; i < nbatch_; ++i) {
      envs_.push_back(MakeRuntime());
    }
  }

  int nbatch() const { return nbatch_; }
  int state_dim() const { return state_dim_; }
  int control_dim() const { return nu_; }
  int nthread() const { return nthread_; }
  int num_filtered_geometries() const { return num_filtered_geometries_; }

  py::dict step(py::array_t<double, py::array::c_style | py::array::forcecast> state0,
                int nstep,
                py::array_t<double, py::array::c_style | py::array::forcecast> control,
                py::object push_force) {
    if (nstep < 1) {
      throw std::invalid_argument("nstep must be >= 1");
    }
    auto state_info = state0.request();
    RequireShape(state_info, {nbatch_, state_dim_}, "state0");
    auto control_info = control.request();
    const bool control_is_traj = control_info.ndim == 3;
    if (control_is_traj) {
      RequireShape(control_info, {nbatch_, nstep, nu_}, "control");
    } else {
      RequireShape(control_info, {nbatch_, nu_}, "control");
    }

    py::array_t<double, py::array::c_style | py::array::forcecast> push_array;
    bool has_push = !push_force.is_none();
    if (has_push) {
      push_array = py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(
          push_force);
      RequireShape(push_array.request(), {nbatch_, 3}, "push_force");
    }

    auto state_out = MakeArray({nbatch_, state_dim_});
    py::dict sensor;
    auto gyro = MakeArray({nbatch_, 3});
    auto local_linvel = MakeArray({nbatch_, 3});
    auto global_linvel = MakeArray({nbatch_, 3});
    auto global_angvel = MakeArray({nbatch_, 3});
    auto upvector = MakeArray({nbatch_, 3});
    auto base_pos = MakeArray({nbatch_, 3});
    auto base_quat = MakeArray({nbatch_, 4});
    auto dof_pos = MakeArray({nbatch_, nu_});
    auto dof_vel = MakeArray({nbatch_, nu_});
    auto feet_pos = MakeArray({nbatch_, static_cast<int>(foot_bodies_.size()), 3});
    auto feet_contact_force = MakeArray({nbatch_, static_cast<int>(foot_bodies_.size()), 3});

    auto start = std::chrono::steady_clock::now();
    {
      py::gil_scoped_release release;
      auto worker = [&](int begin, int end) {
        for (int env_index = begin; env_index < end; ++env_index) {
          StepOne(env_index, state0, control, control_is_traj, nstep,
                  has_push ? &push_array : nullptr);
          WriteState(env_index, state_out);
          WriteSensors(env_index, gyro, local_linvel, global_linvel, global_angvel, upvector,
                       base_pos, base_quat, dof_pos, dof_vel, feet_pos, feet_contact_force);
        }
      };
      RunChunks(worker);
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    const double step_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(elapsed).count();

    sensor["gyro"] = gyro;
    sensor["local_linvel"] = local_linvel;
    sensor["upvector"] = upvector;
    sensor["base_pos"] = base_pos;
    sensor["base_quat"] = base_quat;
    sensor["global_linvel"] = global_linvel;
    sensor["global_angvel"] = global_angvel;
    sensor["position"] = base_pos;
    sensor["dof_pos"] = dof_pos;
    sensor["dof_vel"] = dof_vel;
    sensor["feet_pos"] = feet_pos;
    sensor["feet_contact_force"] = feet_contact_force;
    py::dict timing;
    timing["step_ms"] = step_ms;
    py::dict output;
    output["state"] = state_out;
    output["sensor"] = sensor;
    output["timing"] = timing;
    return output;
  }

  py::dict reset(py::array_t<int, py::array::c_style | py::array::forcecast> env_ids,
                 py::array_t<double, py::array::c_style | py::array::forcecast> initial_state) {
    auto ids_info = env_ids.request();
    if (ids_info.ndim != 1) {
      throw std::invalid_argument("env_ids must be one-dimensional");
    }
    const int rows = static_cast<int>(ids_info.shape[0]);
    RequireShape(initial_state.request(), {rows, state_dim_}, "initial_state");
    {
      py::gil_scoped_release release;
      auto ids = env_ids.unchecked<1>();
      auto state = initial_state.unchecked<2>();
      for (int row = 0; row < rows; ++row) {
        const int env_index = ids(row);
        if (env_index < 0 || env_index >= nbatch_) {
          throw std::out_of_range("env_id out of range");
        }
        LoadState(env_index, &state(row, 0));
      }
    }
    return Snapshot();
  }

  py::dict snapshot() { return Snapshot(); }

 private:
  EnvRuntime MakeRuntime() {
    EnvRuntime runtime;
    auto context = diagram_->CreateDefaultContext();
    runtime.plant_context = &plant_->GetMyMutableContextFromRoot(context.get());
    plant_->get_actuation_input_port(model_instance_)
        .FixValue(runtime.plant_context, Eigen::VectorXd::Zero(nu_));
    SetNativePdTarget(Eigen::VectorXd::Zero(nu_), runtime.plant_context);
    runtime.simulator = std::make_unique<Simulator<double>>(*diagram_, std::move(context));
    runtime.plant_context =
        &plant_->GetMyMutableContextFromRoot(&runtime.simulator->get_mutable_context());
    runtime.simulator->set_target_realtime_rate(0.0);
    runtime.simulator->Initialize();
    return runtime;
  }

  void LoadState(int env_index, const double* state_row) {
    auto& runtime = envs_.at(env_index);
    runtime.simulator->get_mutable_context().SetTime(state_row[0]);
    plant_->SetPositions(runtime.plant_context, MujocoQposToDrake(state_row + 1, nq_));
    plant_->SetVelocities(runtime.plant_context, MujocoQvelToDrake(state_row + 1 + nq_, nv_));
    if (nu_ > 0) {
      Eigen::VectorXd target = Eigen::Map<const Eigen::VectorXd>(state_row + 1 + kRootQposDim, nu_);
      SetNativePdTarget(target, runtime.plant_context);
    }
    runtime.simulator->Initialize();
  }

  void StepOne(int env_index,
               const py::array_t<double, py::array::c_style | py::array::forcecast>& state0,
               const py::array_t<double, py::array::c_style | py::array::forcecast>& control,
               bool control_is_traj, int nstep,
               const py::array_t<double, py::array::c_style | py::array::forcecast>* push_force) {
    auto state = state0.unchecked<2>();
    LoadState(env_index, &state(env_index, 0));
    auto& runtime = envs_.at(env_index);
    auto ctrl_limits = ctrl_limits_.unchecked<2>();
    for (int substep = 0; substep < nstep; ++substep) {
      Eigen::VectorXd target(nu_);
      if (control_is_traj) {
        auto control_values = control.unchecked<3>();
        for (int j = 0; j < nu_; ++j) {
          target[j] = std::clamp(control_values(env_index, substep, j), ctrl_limits(j, 0),
                                 ctrl_limits(j, 1));
        }
      } else {
        auto control_values = control.unchecked<2>();
        for (int j = 0; j < nu_; ++j) {
          target[j] = std::clamp(control_values(env_index, j), ctrl_limits(j, 0),
                                 ctrl_limits(j, 1));
        }
      }
      SetNativePdTarget(target, runtime.plant_context);
      if (push_force != nullptr) {
        auto push_values = push_force->unchecked<2>();
        SetExternalPushForce(env_index, push_values(env_index, 0), push_values(env_index, 1),
                             push_values(env_index, 2));
      } else {
        SetExternalPushForce(env_index, 0.0, 0.0, 0.0);
      }
      runtime.simulator->AdvanceTo(runtime.simulator->get_context().get_time() + sim_dt_);
    }
  }

  void SetNativePdTarget(const Eigen::VectorXd& target_q, Context<double>* plant_context) {
    Eigen::VectorXd desired(2 * nu_);
    desired.head(nu_) = target_q;
    desired.tail(nu_).setZero();
    plant_->get_desired_state_input_port(model_instance_).FixValue(plant_context, desired);
  }

  void SetExternalPushForce(int env_index, double fx, double fy, double fz) {
    auto& runtime = envs_.at(env_index);
    std::vector<ExternallyAppliedSpatialForce<double>> forces;
    if (std::abs(fx) > 0.0 || std::abs(fy) > 0.0 || std::abs(fz) > 0.0) {
      ExternallyAppliedSpatialForce<double> applied;
      applied.body_index = push_body_->index();
      applied.p_BoBq_B.setZero();
      applied.F_Bq_W = SpatialForce<double>(Eigen::Vector3d::Zero(), Eigen::Vector3d(fx, fy, fz));
      forces.push_back(applied);
    }
    plant_->get_applied_spatial_force_input_port().FixValue(runtime.plant_context, forces);
  }

  void WriteState(int env_index, py::array_t<double>& state_out) const {
    auto state = state_out.mutable_unchecked<2>();
    const auto& runtime = envs_.at(env_index);
    state(env_index, 0) = runtime.simulator->get_context().get_time();
    Eigen::VectorXd q = plant_->GetPositions(*runtime.plant_context);
    Eigen::VectorXd v = plant_->GetVelocities(*runtime.plant_context);
    DrakeQposToMujoco(q, &state(env_index, 1));
    DrakeQvelToMujoco(v, &state(env_index, 1 + nq_));
  }

  void WriteSensors(int env_index, py::array_t<double>& gyro, py::array_t<double>& local_linvel,
                    py::array_t<double>& global_linvel, py::array_t<double>& global_angvel,
                    py::array_t<double>& upvector, py::array_t<double>& base_pos,
                    py::array_t<double>& base_quat, py::array_t<double>& dof_pos,
                    py::array_t<double>& dof_vel, py::array_t<double>& feet_pos,
                    py::array_t<double>& feet_contact_force) const {
    const auto& runtime = envs_.at(env_index);
    const RigidTransform<double> x_wb =
        plant_->EvalBodyPoseInWorld(*runtime.plant_context, *trunk_body_);
    const Eigen::Matrix3d r_bw = x_wb.rotation().matrix().transpose();
    const auto velocity_w =
        plant_->EvalBodySpatialVelocityInWorld(*runtime.plant_context, *trunk_body_);
    const Eigen::Vector3d gyro_b = r_bw * velocity_w.rotational();
    const Eigen::Vector3d linvel_b = r_bw * velocity_w.translational();
    auto gyro_view = gyro.mutable_unchecked<2>();
    auto local_linvel_view = local_linvel.mutable_unchecked<2>();
    auto global_linvel_view = global_linvel.mutable_unchecked<2>();
    auto global_angvel_view = global_angvel.mutable_unchecked<2>();
    auto upvector_view = upvector.mutable_unchecked<2>();
    auto base_pos_view = base_pos.mutable_unchecked<2>();
    auto base_quat_view = base_quat.mutable_unchecked<2>();
    for (int i = 0; i < 3; ++i) {
      gyro_view(env_index, i) = gyro_b[i];
      local_linvel_view(env_index, i) = linvel_b[i];
      global_linvel_view(env_index, i) = velocity_w.translational()[i];
      global_angvel_view(env_index, i) = velocity_w.rotational()[i];
      upvector_view(env_index, i) = x_wb.rotation().matrix()(i, 2);
      base_pos_view(env_index, i) = x_wb.translation()[i];
    }
    const auto quat = x_wb.rotation().ToQuaternion();
    base_quat_view(env_index, 0) = quat.w();
    base_quat_view(env_index, 1) = quat.x();
    base_quat_view(env_index, 2) = quat.y();
    base_quat_view(env_index, 3) = quat.z();

    Eigen::VectorXd q = plant_->GetPositions(*runtime.plant_context);
    Eigen::VectorXd v = plant_->GetVelocities(*runtime.plant_context);
    auto dof_pos_view = dof_pos.mutable_unchecked<2>();
    auto dof_vel_view = dof_vel.mutable_unchecked<2>();
    for (int i = 0; i < nu_; ++i) {
      dof_pos_view(env_index, i) = q[kRootQposDim + i];
      dof_vel_view(env_index, i) = v[kRootQvelDim + i];
    }

    auto foot_offsets = foot_offsets_.unchecked<2>();
    auto feet_pos_view = feet_pos.mutable_unchecked<3>();
    auto feet_contact_force_view = feet_contact_force.mutable_unchecked<3>();
    for (int foot = 0; foot < static_cast<int>(foot_bodies_.size()); ++foot) {
      const RigidTransform<double> x_wf =
          plant_->EvalBodyPoseInWorld(*runtime.plant_context, *foot_bodies_[foot]);
      const Eigen::Vector3d offset(foot_offsets(foot, 0), foot_offsets(foot, 1),
                                   foot_offsets(foot, 2));
      const Eigen::Vector3d p = x_wf.translation() + x_wf.rotation().matrix() * offset;
      for (int axis = 0; axis < 3; ++axis) {
        feet_pos_view(env_index, foot, axis) = p[axis];
        feet_contact_force_view(env_index, foot, axis) = 0.0;
      }
    }
  }

  py::dict Snapshot() {
    auto state_out = MakeArray({nbatch_, state_dim_});
    auto gyro = MakeArray({nbatch_, 3});
    auto local_linvel = MakeArray({nbatch_, 3});
    auto global_linvel = MakeArray({nbatch_, 3});
    auto global_angvel = MakeArray({nbatch_, 3});
    auto upvector = MakeArray({nbatch_, 3});
    auto base_pos = MakeArray({nbatch_, 3});
    auto base_quat = MakeArray({nbatch_, 4});
    auto dof_pos = MakeArray({nbatch_, nu_});
    auto dof_vel = MakeArray({nbatch_, nu_});
    auto feet_pos = MakeArray({nbatch_, static_cast<int>(foot_bodies_.size()), 3});
    auto feet_contact_force = MakeArray({nbatch_, static_cast<int>(foot_bodies_.size()), 3});
    for (int env_index = 0; env_index < nbatch_; ++env_index) {
      WriteState(env_index, state_out);
      WriteSensors(env_index, gyro, local_linvel, global_linvel, global_angvel, upvector, base_pos,
                   base_quat, dof_pos, dof_vel, feet_pos, feet_contact_force);
    }
    py::dict sensor;
    sensor["gyro"] = gyro;
    sensor["local_linvel"] = local_linvel;
    sensor["upvector"] = upvector;
    sensor["base_pos"] = base_pos;
    sensor["base_quat"] = base_quat;
    sensor["global_linvel"] = global_linvel;
    sensor["global_angvel"] = global_angvel;
    sensor["dof_pos"] = dof_pos;
    sensor["dof_vel"] = dof_vel;
    sensor["feet_pos"] = feet_pos;
    sensor["feet_contact_force"] = feet_contact_force;
    py::dict output;
    output["state"] = state_out;
    output["sensor"] = sensor;
    output["timing"] = py::dict();
    return output;
  }

  int ExcludeRobotSelfCollisions() {
    std::vector<const RigidBody<double>*> bodies;
    for (BodyIndex body_index : plant_->GetBodyIndices(model_instance_)) {
      bodies.push_back(&plant_->get_body(body_index));
    }
    int count = 0;
    for (const RigidBody<double>* body : bodies) {
      count += static_cast<int>(plant_->GetCollisionGeometriesForBody(*body).size());
    }
    if (count > 0) {
      GeometrySet robot_geometries = plant_->CollectRegisteredGeometries(bodies);
      scene_graph_->collision_filter_manager().Apply(
          CollisionFilterDeclaration().ExcludeWithin(std::move(robot_geometries)));
    }
    return count;
  }

  template <typename Worker>
  void RunChunks(Worker worker) {
    const int thread_count = std::min(nthread_, nbatch_);
    if (thread_count <= 1) {
      worker(0, nbatch_);
      return;
    }
    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    for (int thread = 0; thread < thread_count; ++thread) {
      const int begin = thread * nbatch_ / thread_count;
      const int end = (thread + 1) * nbatch_ / thread_count;
      threads.emplace_back([&, begin, end]() { worker(begin, end); });
    }
    for (auto& thread : threads) {
      thread.join();
    }
  }

  int nbatch_{};
  double sim_dt_{};
  int nq_{};
  int nv_{};
  int nu_{};
  int state_dim_{};
  py::array_t<double> ctrl_limits_;
  py::array_t<double> torque_limits_;
  std::vector<int> foot_body_indices_;
  py::array_t<double> foot_offsets_;
  double kp_{};
  double kd_{};
  int nthread_{};
  int num_filtered_geometries_{};
  std::unique_ptr<Diagram<double>> diagram_;
  MultibodyPlant<double>* plant_{};
  SceneGraph<double>* scene_graph_{};
  ModelInstanceIndex model_instance_;
  const RigidBody<double>* trunk_body_{};
  const RigidBody<double>* push_body_{};
  std::vector<const RigidBody<double>*> foot_bodies_;
  std::vector<EnvRuntime> envs_;
};

bool NativeAvailable() { return true; }

}  // namespace

PYBIND11_MODULE(_drake_env_pool, m) {
  m.doc() = "Optional native DrakeEnvPool for UniLab Go1 DrakeUni experiments.";
  m.def("native_available", &NativeAvailable);
  py::class_<NativeDrakeEnvPool>(m, "NativeDrakeEnvPool")
      .def(py::init<const std::string&, int, double,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    py::array_t<double, py::array::c_style | py::array::forcecast>, int, int,
                    const std::vector<int>&,
                    py::array_t<double, py::array::c_style | py::array::forcecast>, double,
                    double, int>(),
           py::arg("model_file"), py::arg("nbatch"), py::arg("sim_dt"),
           py::arg("ctrl_limits"), py::arg("torque_limits"), py::arg("base_body_index"),
           py::arg("push_body_index"), py::arg("foot_body_indices"), py::arg("foot_offsets"),
           py::arg("kp"), py::arg("kd"), py::arg("nthread") = 1)
      .def_property_readonly("nbatch", &NativeDrakeEnvPool::nbatch)
      .def_property_readonly("state_dim", &NativeDrakeEnvPool::state_dim)
      .def_property_readonly("control_dim", &NativeDrakeEnvPool::control_dim)
      .def_property_readonly("nthread", &NativeDrakeEnvPool::nthread)
      .def_property_readonly("num_filtered_geometries",
                             &NativeDrakeEnvPool::num_filtered_geometries)
      .def("step", &NativeDrakeEnvPool::step, py::arg("state0"), py::arg("nstep"),
           py::arg("control"), py::arg("push_force") = py::none())
      .def("reset", &NativeDrakeEnvPool::reset, py::arg("env_ids"), py::arg("initial_state"))
      .def("snapshot", &NativeDrakeEnvPool::snapshot);
}
