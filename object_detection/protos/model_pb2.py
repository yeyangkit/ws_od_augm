# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: object_detection/protos/model.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from object_detection.protos import ssd_pb2 as object__detection_dot_protos_dot_ssd__pb2
from object_detection.protos import ssd_augmentation_pb2 as object__detection_dot_protos_dot_ssd__augmentation__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='object_detection/protos/model.proto',
  package='object_detection.protos',
  syntax='proto2',
  serialized_pb=_b('\n#object_detection/protos/model.proto\x12\x17object_detection.protos\x1a!object_detection/protos/ssd.proto\x1a.object_detection/protos/ssd_augmentation.proto\"\xbc\x01\n\x0e\x44\x65tectionModel\x12+\n\x03ssd\x18\x01 \x01(\x0b\x32\x1c.object_detection.protos.SsdH\x00\x12\x44\n\x10ssd_augmentation\x18\x02 \x01(\x0b\x32(.object_detection.protos.SsdAugmentationH\x00\x12\x16\n\x0einput_features\x18\x03 \x03(\t\x12\x16\n\x0einput_channels\x18\x04 \x03(\x05\x42\x07\n\x05model')
  ,
  dependencies=[object__detection_dot_protos_dot_ssd__pb2.DESCRIPTOR,object__detection_dot_protos_dot_ssd__augmentation__pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_DETECTIONMODEL = _descriptor.Descriptor(
  name='DetectionModel',
  full_name='object_detection.protos.DetectionModel',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='ssd', full_name='object_detection.protos.DetectionModel.ssd', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='ssd_augmentation', full_name='object_detection.protos.DetectionModel.ssd_augmentation', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='input_features', full_name='object_detection.protos.DetectionModel.input_features', index=2,
      number=3, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='input_channels', full_name='object_detection.protos.DetectionModel.input_channels', index=3,
      number=4, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='model', full_name='object_detection.protos.DetectionModel.model',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=148,
  serialized_end=336,
)

_DETECTIONMODEL.fields_by_name['ssd'].message_type = object__detection_dot_protos_dot_ssd__pb2._SSD
_DETECTIONMODEL.fields_by_name['ssd_augmentation'].message_type = object__detection_dot_protos_dot_ssd__augmentation__pb2._SSDAUGMENTATION
_DETECTIONMODEL.oneofs_by_name['model'].fields.append(
  _DETECTIONMODEL.fields_by_name['ssd'])
_DETECTIONMODEL.fields_by_name['ssd'].containing_oneof = _DETECTIONMODEL.oneofs_by_name['model']
_DETECTIONMODEL.oneofs_by_name['model'].fields.append(
  _DETECTIONMODEL.fields_by_name['ssd_augmentation'])
_DETECTIONMODEL.fields_by_name['ssd_augmentation'].containing_oneof = _DETECTIONMODEL.oneofs_by_name['model']
DESCRIPTOR.message_types_by_name['DetectionModel'] = _DETECTIONMODEL

DetectionModel = _reflection.GeneratedProtocolMessageType('DetectionModel', (_message.Message,), dict(
  DESCRIPTOR = _DETECTIONMODEL,
  __module__ = 'object_detection.protos.model_pb2'
  # @@protoc_insertion_point(class_scope:object_detection.protos.DetectionModel)
  ))
_sym_db.RegisterMessage(DetectionModel)


# @@protoc_insertion_point(module_scope)
